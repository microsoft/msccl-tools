# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from lxml import etree as ET
from collections import defaultdict
from dataclasses import dataclass, field, replace
import math
import threading, queue, itertools, bisect
from enum import Enum
from z3 import *

from sccl.language.ir import *
from sccl.language import *
import sccl.language.collectives as lang_collectives

class ChannelPolicy(Enum):
    One = 'One'
    MatchTopology = 'MatchTopology'

    def __str__(self):
        return self.value

@dataclass
class CopyOp:
    src_buf: Buffer
    src_off: int
    dst_buf: Buffer
    dst_off: int
    cnt: int

def get_buffer_and_offset(gpu, addr):
    # Map an address to one of the named buffers
    if addr in gpu.inputs:
        return Buffer.input, gpu.inputs[addr]
    elif addr in gpu.outputs:
        return Buffer.output, gpu.outputs[addr]
    elif addr in gpu.scratch:
        return Buffer.scratch, gpu.scratch[addr]
    else:
        raise RuntimeError('Address is not mapped to a buffer')

def _analyze_liveness(gpus, algorithm):
    # Initialize liveness intervals for buffers on each GPU
    input_livenesses = {rank: [[(-1,-1)] for _ in range(gpu.input_chunks)] for rank, gpu in gpus.items()}
    output_livenesses = {rank: [[(math.inf,math.inf)] for _ in range(gpu.output_chunks)] for rank, gpu in gpus.items()}
    scratch_livenesses = {rank: [[(math.inf,-1)] for addr, idx in gpu.scratch.items()] for rank, gpu in gpus.items()}

    def update_liveness(rank, addr, step_idx):
        gpu = gpus[rank]
        # Find the relevant buffer and livenesses for the address
        # Addresses in both input and output are treated as input (as currently postcopies are inserted).
        # TODO: This is a bit dangerous, as changing the other bit of code to do precopies would silently break this.
        if addr in gpu.inputs:
            buffer = gpu.inputs
            liveness = input_livenesses[rank]
        elif addr in gpu.outputs:
            buffer = gpu.outputs
            liveness = output_livenesses[rank]
        elif addr in gpu.scratch:
            buffer = gpu.scratch
            liveness = scratch_livenesses[rank]
        else:
            raise RuntimeError(f'Address {addr} not found in any buffer of rank {rank}.')
        
        # Expand the interval to include the step
        idx = buffer[addr]
        start, end = liveness[idx][0]
        liveness[idx][0] = (min(start, step_idx), max(end, step_idx))

    # For each step of the algorithm, update liveness intervals for all buffers
    for step_idx, step in enumerate(algorithm.steps):
        for addr, src, dst in step.sends:
            update_liveness(src, addr, step_idx)
            update_liveness(dst, addr, step_idx)
    
    return (input_livenesses, output_livenesses, scratch_livenesses)

def _remap_scratch_into_input_output(liveness, gpus, logging):
    '''
    This function solves and applies a static mapping for scratch buffer indices to input/output buffers that minimizes
    scratch buffer usage for each GPU. The solving is done per GPU using the Z3 SMT solver.
    '''
    input_livenesses, output_livenesses, scratch_livenesses = liveness

    if logging:
        print('Remapping scratch into input/output...')

    def conflict(b1, b2):
        # Check if any of the intervals in lists b1 and b2 overlap
        return any(s1 <= e2 and s2 <= e1 for s1, e1 in b1 for s2, e2 in b2)

    print('Optimizing scratch mapping on all GPUs: ', end='', flush=True)
    # Handle each GPU separately
    for rank, gpu in gpus.items():
        ctx = Context()
        s = Solver(ctx=ctx)

        def remap(idx):
            # Choose for each scratch index a new index in one of the buffers
            # The index space has the input buffer from 0 to input_chunks-1,
            # the output buffer from input_chunks to output_chunks-1,
            # and the scratch buffer for any indices past that.
            return Int(f'{idx}_remap', ctx=ctx)

        # This variable limits the maximum index, in effect the size of the scratch buffer
        idx_end = Int(f'idx_end', ctx=ctx)

        for scratch_idx, scratch_liveness in enumerate(scratch_livenesses[rank]):
            # Block any input indices that conflict with the scratch index
            for input_idx, liveness in enumerate(input_livenesses[rank]):
                if conflict(scratch_liveness, liveness):
                    s.add(remap(scratch_idx) != input_idx)
            # Block any output indices that conflict with the scratch index
            for output_idx, liveness in enumerate(output_livenesses[rank]):
                if conflict(scratch_liveness, liveness):
                    s.add(remap(scratch_idx) != output_idx + gpu.input_chunks)
            # Block remapping conflicting scratch indices to the same input/output indices
            for other_idx, liveness in enumerate(scratch_livenesses[rank]):
                if other_idx != scratch_idx and conflict(liveness, scratch_liveness):
                    s.add(remap(scratch_idx) != remap(other_idx))
            # Require all indices to fit in the allowed buffer space
            s.add(remap(scratch_idx) >= 0)
            s.add(remap(scratch_idx) < idx_end)

        no_memory = gpu.input_chunks + gpu.output_chunks

        q = queue.Queue()
        def optimize(q):
            # Iterate the memory limit down to find a mapping that minimizes scratch usage
            for memory in range(no_memory + gpu.scratch_size(), no_memory - 1, -1):
                if s.check(idx_end == memory) == sat:
                    # Remember the model for the best solution
                    try:
                        m = s.model()
                        new_idxs = {addr: m[remap(old_idx)].as_long() for addr, old_idx in gpu.scratch.items()}
                        q.put(new_idxs)
                    except Z3Exception:
                        # This can happen when the solver is interrupted
                        return
                else:
                    return
        t = threading.Thread(target=optimize, args=(q,))
        t.start()
        t.join(1)
        ctx.interrupt()

        new_idxs = None
        while not q.empty():
            new_idxs = q.get()
        
        if new_idxs != None:
            print('.', end='', flush=True)
            # Apply the model to remap the scratch indices
            new_scratch = {}
            new_scratch_livenesses = [[] for addr, idx in gpu.scratch.items()]
            for addr, old_idx in gpu.scratch.items():
                new_idx = new_idxs[addr]
                # Figure out which buffer the index is in
                if new_idx < gpu.input_chunks:
                    tgt_buffer = gpu.inputs
                    tgt_idx = new_idx
                    tgt_liveness = input_livenesses[rank][tgt_idx]
                elif new_idx < gpu.input_chunks + gpu.output_chunks:
                    tgt_buffer = gpu.outputs
                    tgt_idx = new_idx - gpu.input_chunks
                    tgt_liveness = output_livenesses[rank][tgt_idx]
                else:
                    tgt_buffer = new_scratch
                    tgt_idx = new_idx - gpu.input_chunks - gpu.output_chunks
                    tgt_liveness = new_scratch_livenesses[tgt_idx]

                # Check that the remapping doesn't conflict with any existing mappings
                liveness = scratch_livenesses[rank][old_idx]
                assert not conflict(tgt_liveness, liveness)
                tgt_liveness.extend(liveness)

                # Remap the scratch index to the new index in the target buffer
                tgt_buffer[addr] = tgt_idx
            gpu.scratch = new_scratch
        else:
            print('x', end='', flush=True)
    else:
        print()

    if logging:
        max_scratch_overhead = max(gpu.scratch_size() / (gpu.input_chunks + gpu.output_chunks) for gpu in gpus.values())
        print(f'Maximum scratch overhead is {max_scratch_overhead * 100:.0f}%')

def _greedy_scratch_sort(algorithm, gpus):
    # Sort scratch mappings in an attempt to make more of them contiguous (this is of course a heuristic).
    # The procedure first figures out the sets of addresses that would result in combined operations if
    # the source and destination indices were contiguously allocated. These are then greedily allocated
    # starting with the largest sets. Afterwards any remaining scratch mappings are allocated in order.
    tosort = { rank: set(gpu.scratch.keys()).union(gpu.inputs.keys()).union(gpu.outputs.keys()) for rank, gpu in gpus.items() }
    csets = defaultdict(set)
    for idx, step in enumerate(algorithm.steps):
        for addr, src, dst in step.sends:
            if addr in tosort[src] and addr in tosort[dst]:
                csets[(idx, src, dst)].add(addr)
    for gpu in gpus.values():
        gpu.scratch = {}
    for key in sorted(csets, key=lambda x: len(csets[x]), reverse=True):
        idx, src, dst = key
        cset = csets[key]

        def contiguous_in(buffer):
            if not cset.issubset(buffer.keys()):
                return False
            for i in range(1, len(addrs)):
                if buffer[addrs[i]] != buffer[addrs[i-1]] + 1:
                    return False
            return True
        
        # Check if either side is already contiguous
        addrs = sorted(cset)
        src_input_contig = contiguous_in(gpus[src].inputs)
        skip_src = src_input_contig or contiguous_in(gpus[src].outputs) or contiguous_in(gpus[src].scratch)
        dst_input_contig = contiguous_in(gpus[dst].inputs) 
        skip_dst = dst_input_contig or contiguous_in(gpus[dst].outputs) or contiguous_in(gpus[dst].scratch)

        if (cset.issubset(tosort[src]) or skip_src) and (cset.issubset(tosort[dst]) or skip_dst):
            # Block these addresses from being sorted again on both GPUs
            tosort[src].difference_update(cset)
            tosort[dst].difference_update(cset)

            for addr in addrs:
                def alloc(rank, skip, prefer_input):
                    gpu = gpus[rank]
                    if skip:
                        # If not allocating in scratch, check if we need to make a copy and do a precopy if that allows
                        # maintaining contiguity.
                        if addr in gpu.inputs and addr in gpu.outputs:
                            op = CopyOp(Buffer.input, gpu.inputs[addr], Buffer.output, gpu.outputs[addr], 1)
                            if prefer_input:
                                gpu.postcopies.append(op)
                                del gpu.outputs[addr]
                            else:
                                gpu.precopies.append(op)
                                del gpu.inputs[addr]
                    else:
                        # Reallocate address in scratch and insert necessary copies for input/output addresses
                        gpu.scratch[addr] = len(gpu.scratch)
                        if addr in gpu.inputs:
                            op = CopyOp(Buffer.input, gpu.inputs[addr], Buffer.scratch, gpu.scratch[addr], 1)
                            gpu.precopies.append(op)
                            del gpu.inputs[addr]
                        if addr in gpu.outputs:
                            op = CopyOp(Buffer.scratch, gpu.scratch[addr], Buffer.output, gpu.outputs[addr], 1)
                            gpu.postcopies.append(op)
                            del gpu.outputs[addr]
                alloc(src, skip_src, src_input_contig)
                alloc(dst, skip_dst, dst_input_contig)

    # Allocate any remaining addresses that aren't already input or output
    for rank in tosort:
        gpu = gpus[rank]
        for addr in sorted(tosort[rank]):
            if not addr in gpu.inputs and not addr in gpu.outputs:
                gpu.scratch[addr] = len(gpu.scratch)

def instance_metadata(gpus, instances):
    for rank, gpu in gpus.items():
            # Multiply metadata with instances
            def expand_mappings(mappings):
                return { addr * instances + i: idx * instances + i for addr, idx in mappings.items() for i in range(instances) }
            gpu.inputs = expand_mappings(gpu.inputs)
            gpu.outputs = expand_mappings(gpu.outputs)
            gpu.input_chunks *= instances
            gpu.output_chunks *= instances
            gpu.scratch = expand_mappings(gpu.scratch)

def ncclize(algorithm, remap_scratch = None, channel_policy=ChannelPolicy.MatchTopology, pretty_print = True, 
        use_scratch=True, merge_contiguous=True, greedy_scratch_sorting=False, instances=1, logging=False,
        instr_fusion=True):
    '''
    Generate the XML format used by the NCCL SCCL backend.

    Sends are split into send/recv operations and grouped by the rank executing them. Within each rank operations are
    grouped under <threadblock/> tags, which handle 1) a single peer, 2) a single type of operation, and 3) at most one
    operation per each step of the algorithm. Additional threadblocks are created as necessary to meet these
    constraints.

    Each send operation is mapped from the abstract addresses used by the synthesized algorithm to offsets into three
    named buffers, "input", "output" and "scratch", based on whether the address appears in a particular rank's
    precondition, postcondition or neither. For addresses that would be in both the input and output buffers <copy/>
    tags are created to mark an initial transfer to the output buffer and only the output buffer mapping is kept.
    '''

    if algorithm.is_pipelined():
        raise ValueError('Pipelining is not supported.')

    if remap_scratch is None:
        if algorithm.instance.extra_memory != None:
            remap_scratch = True
            if logging:
                print('Turning scratch remapping on to honor the memory limit set in the instance.')
        else:
            remap_scratch = False

    # Create GPUs, their address to buffer mappings and possible copies
    gpus = {}
    for rank in algorithm.ranks():
        outputs = {}
        if rank in algorithm.output_map:
            outputs.update({ addr: idx for idx, addr in enumerate(sorted(algorithm.output_map[rank])) })
        inputs = {}
        if rank in algorithm.input_map:
            inputs.update({ addr: idx for idx, addr in enumerate(sorted(algorithm.input_map[rank])) })
        gpus[rank] = Gpu(rank, [], [], [], inputs, outputs, len(inputs), len(outputs))

    # Create scratch buffer mappings if necessary
    def allocate_scratch(gpu, addr):
        if not (addr in gpu.inputs or addr in gpu.outputs or addr in gpu.scratch):
            offset = len(gpu.scratch)
            gpu.scratch[addr] = offset
    for step in algorithm.steps:
        for addr, src, dst in step.sends:
            allocate_scratch(gpus[src], addr)
            allocate_scratch(gpus[dst], addr)

    if remap_scratch:
        # Analyze liveness of indices in buffers and remap scratch into input/output as possible
        liveness = _analyze_liveness(gpus, algorithm)
        _remap_scratch_into_input_output(liveness, gpus, logging)
    elif greedy_scratch_sorting:
        _greedy_scratch_sort(algorithm, gpus)
    else:
        # Sort scratch mappings in an attempt to make more of them contiguous (this is of course a heuristic).
        for gpu in gpus.values():
            gpu.scratch = { addr: idx for idx, addr in enumerate(sorted(gpu.scratch)) }

    # Sort and combine contiguous copy operations
    for rank, gpu in gpus.items():
        def combine_copies(copies):
            copies.sort(key=lambda x: (x.src_buf, x.dst_buf, x.src_off, x.dst_off))
            i = 0
            while i < len(copies) - 1:
                c1 = copies[i]
                c2 = copies[i+1]
                if (c1.src_buf == c2.src_buf and c1.dst_buf == c2.dst_buf and
                    c1.src_off + c1.cnt == c2.src_off and c1.dst_off + c1.cnt == c2.dst_off):
                    c1.cnt += c2.cnt
                    del copies[i+1]
                else:
                    i += 1
        combine_copies(gpu.precopies)
        combine_copies(gpu.postcopies)

    ##### Sort of the end of buffer management

    # Expand copies by instances if necessary
    # if instances > 1:
    #     print("here")
    #     for rank, gpu in gpus.items():
    #         for copy in itertools.chain(gpu.precopies, gpu.postcopies):
    #             print("hello")
    #             copy.src_off *= instances
    #             copy.dst_off *= instances
    #             copy.cnt *= instances

    def make_intervals(src, dst, addrs_set):
        if len(addrs_set) == 0:
            return

        buffs_and_offs = []
        for addr in addrs_set:
            srcbuff, srcoff = get_buffer_and_offset(gpus[src], addr)
            dstbuff, dstoff = get_buffer_and_offset(gpus[dst], addr)
            buffs_and_offs.append((srcbuff, srcoff, dstbuff, dstoff))
        
        if merge_contiguous:
            # Sort sends by both buffers and offsets and merge sends into larger intervals when both the source and
            # destination are contiguous.
            buffs_and_offs.sort()
            start = prev = buffs_and_offs[0]

            def make_interval(a,b):
                cnt = b[1] - a[1] + 1
                assert cnt == b[3] - a[3] + 1, 'Source and destination count mismatch'
                return (a[0], a[1], a[2], a[3], cnt)
        
            for x in buffs_and_offs[1:]:
                if x[0] == prev[0] and x[1] == prev[1] + 1 and x[2] == prev[2] and x[3] == prev[3] + 1:
                    # Merge into previous interval if buffers match and the new offsets are at the end of the interval
                    prev = x
                else:
                    # Yield the previous interval and start a new one
                    yield make_interval(start, prev)
                    start = prev = x
            # Yield the last interval
            yield make_interval(start, prev)
        else:
            # Just yield size 1 intervals if merging is disabled
            for srcbuff, srcoff, dstbuff, dstoff in buffs_and_offs:
                yield (srcbuff, srcoff, dstbuff, dstoff, 1)    

    # Turn all steps of the algorithm into operations
    ops_by_channel = defaultdict(list)

    sends_by_step = []
    for step_idx, step in enumerate(algorithm.steps):
        # Group sent addresses by edge
        grouped_sends = defaultdict(set)
        for addr, src, dst in step.sends:
            grouped_sends[(src,dst)].add(addr)

        # Combine sends into intervals and create multiple instances if necessary
        sends = []
        for (src, dst), addrs in grouped_sends.items():
            intervals = list(make_intervals(src, dst, addrs))
            if channel_policy == ChannelPolicy.One:
                num_chans = 1
                channeled_intervals = [ (src_buf, src_off, dst_buf, dst_off, cnt, 0) for src_buf, src_off, dst_buf, dst_off, cnt in intervals ]
            elif channel_policy == ChannelPolicy.MatchTopology:
                # Divide sends onto channels matching the topology (assume bw is ideal concurrency)
                # Sends are split to balance channels if necessary
                num_chans = algorithm.topology.link(src,dst)
                channeled_intervals = []

                intervals.sort(key=lambda x: x[4])
                counts = [x[4] for x in intervals]
                total = sum(counts)
                targets = [(total//num_chans) + (1 if i < (total%num_chans) else 0) for i in range(num_chans)]

                chan = 0
                while len(intervals) > 0:
                    if targets[chan] >= counts[-1]:
                        i = -1
                    else:
                        i = bisect.bisect_left(counts, targets[chan])
                        if i == len(counts) or counts[i] != targets[chan]:
                            i = -1
                    src_buf, src_off, dst_buf, dst_off, cnt = intervals[i]
                    del intervals[i]
                    del counts[i]
                    if cnt > targets[chan]:
                        rem = cnt - targets[chan]
                        cnt = targets[chan]
                        j = bisect.bisect_left(counts, rem)
                        intervals.insert(j, (src_buf, src_off + cnt, dst_buf, dst_off + cnt, rem))
                        counts.insert(j, rem)

                    channeled_intervals.append((src_buf, src_off, dst_buf, dst_off, cnt, chan))
                    targets[chan] -= cnt
                    assert targets[chan] >= 0
                    if targets[chan] == 0:
                        chan += 1
            else:
                assert False, 'Unhandled channel policy'

            for src_buf, src_off, dst_buf, dst_off, cnt, chan in channeled_intervals:
                for i in range(1):
                    new_src_off = src_off * 1 + i * cnt
                    new_dst_off = dst_off * 1 + i * cnt
                    send = (src, dst, src_buf, new_src_off, dst_buf, new_dst_off, cnt, chan * 1 + i)
                    sends.append(send)
        sends_by_step.append(sends)

    # instance_metadata(gpus, instances)

    # Lower into a SCCLang program
    inplace = False
    chunks = algorithm.collective.num_chunks
    co_name = algorithm.collective.runtime_name
    num_ranks = algorithm.topology.num_nodes()
    print(chunks, num_ranks)
    # TODO: Make the collectives for synthesizer and language the same
    if co_name == 'allreduce':
        collective = lang_collectives.AllReduce(num_ranks, chunks, inplace)
    elif co_name == 'allgather':
        collective = lang_collectives.AllGather(num_ranks, chunks // num_ranks, inplace)
    elif co_name == 'alltoall':
        collective = lang_collectives.AllToAll(num_ranks, chunks // num_ranks, inplace)
    elif co_name == 'reduce_scatter':
        collective = lang_collectives.ReduceScatter(num_ranks, chunks, inplace)
    # TODO: SCCLang instances are they equivalent?
    program = SCCLProgram(algorithm.name, algorithm.topology, collective, 1, instr_fusion=instr_fusion)
    with program:
        for rank, gpu in gpus.items():
            for copy_op in gpu.precopies:
                chunk(rank, copy_op.src_buf, copy_op.src_off, copy_op.cnt).send(rank, copy_op.dst_buf, copy_op.dst_off)

        for step_idx, sends in enumerate(sends_by_step):
            # print(step_idx)
            for src, dst, src_buf, src_off, dst_buf, dst_off, cnt, chan in sends:
                # print("  ", src, chan, src_buf, src_off, dst, dst_buf, dst_off, cnt)
                chunk(src, src_buf, src_off, cnt).send(dst, dst_buf, dst_off, ch=chan)

        for rank, gpu in gpus.items():
            for copy_op in gpu.postcopies:
                chunk(rank, copy_op.src_buf, copy_op.src_off, copy_op.cnt).send(rank, copy_op.dst_buf, copy_op.dst_off)

        # Add any copies from input to output that weren't already added
        for rank, gpu in gpus.items():
            for addr in gpu.inputs:
                if addr in gpu.outputs:
                    chunk(rank, Buffer.input, gpu.inputs[addr]).send(rank, Buffer.output, gpu.outputs[addr])
                    del gpu.outputs[addr]
                    
    return ir_to_xml(program.lower())
