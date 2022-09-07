# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from lxml import etree as ET
from collections import defaultdict
from dataclasses import dataclass, field, replace
import math
import threading, queue, itertools, bisect
from enum import Enum
from z3 import *

@dataclass
class _Gpu:
    precopies: list
    postcopies: list
    inputs: dict
    outputs: dict
    input_chunks: int
    output_chunks: int
    scratch: dict = field(default_factory=dict)
    threadblocks: list = field(default_factory=list)

    def scratch_size(self):
        return max((idx for addr, idx in self.scratch.items()), default=-1) + 1

@dataclass
class _Threadblock:
    channel: int
    rbid: int = None
    send: int = -1
    recv: int = -1
    steps: list = field(default_factory=list)
    # The steps may expand into multiple operations here
    ops: list = field(default_factory=list)

@dataclass
class _Op:
    gpu: int
    peer: int
    step: int
    is_send: bool
    op_type: str
    src_buffer: str
    src_offset: int
    dst_buffer: str
    dst_offset: int
    cnt: int
    depends: list
    block_rbid: int = None
    # idx is the NCCL XML step index, which may not be the same as the algorithm step index
    idx: int = None
    has_dependence: bool = False

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

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
                            copy = _Op(rank, None, -1, False, 'cpy', 'i', gpu.inputs[addr], 'o', gpu.outputs[addr], 1, [])
                            if prefer_input:
                                gpu.postcopies.append(copy)
                                del gpu.outputs[addr]
                            else:
                                gpu.precopies.append(copy)
                                del gpu.inputs[addr]
                    else:
                        # Reallocate address in scratch and insert necessary copies for input/output addresses
                        gpu.scratch[addr] = len(gpu.scratch)
                        if addr in gpu.inputs:
                            gpu.precopies.append(_Op(src, None, -1, False, 'cpy',
                                'i', gpu.inputs[addr], 's', gpu.scratch[addr], 1, []))
                            del gpu.inputs[addr]
                        if addr in gpu.outputs:
                            gpu.postcopies.append(_Op(src, None, -1, False, 'cpy',
                                's', gpu.scratch[addr], 'o', gpu.outputs[addr], 1, []))
                            del gpu.outputs[addr]
                alloc(src, skip_src, src_input_contig)
                alloc(dst, skip_dst, dst_input_contig)

    # Allocate any remaining addresses that aren't already input or output
    for rank in tosort:
        gpu = gpus[rank]
        for addr in sorted(tosort[rank]):
            if not addr in gpu.inputs and not addr in gpu.outputs:
                gpu.scratch[addr] = len(gpu.scratch)

class ChannelPolicy(Enum):
    One = 'One'
    MatchTopology = 'MatchTopology'

    def __str__(self):
        return self.value

def ncclize(algorithm, remap_scratch = None, channel_policy=ChannelPolicy.MatchTopology, pretty_print = True, use_scratch=True, merge_contiguous=True, greedy_scratch_sorting=False, instances=1, logging=False):
    '''
    Generate the XML format used by the NCCL MSCCL backend.

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
        gpus[rank] = _Gpu([], [], inputs, outputs, len(inputs), len(outputs))

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
        if algorithm.collective.is_combining:
            raise RuntimeError('Combining collectives are not supported yet with scratch remapping.')
    elif greedy_scratch_sorting:
        _greedy_scratch_sort(algorithm, gpus)
    else:
        # Sort scratch mappings in an attempt to make more of them contiguous (this is of course a heuristic).
        for gpu in gpus.values():
            gpu.scratch = { addr: idx for idx, addr in enumerate(sorted(gpu.scratch)) }

    # Add any copies from input to output that weren't already added
    for rank, gpu in gpus.items():
        for addr in gpu.inputs:
            if addr in gpu.outputs:
                gpu.postcopies.append(_Op(rank, None, -1, False, 'cpy',
                    'i', gpu.inputs[addr], 'o', gpu.outputs[addr], 1, []))
                del gpu.outputs[addr]

    # Sort and combine contiguous copy operations
    for rank, gpu in gpus.items():
        def combine_copies(copies):
            copies.sort(key=lambda x: (x.src_buffer, x.dst_buffer, x.src_offset, x.dst_offset))
            i = 0
            while i < len(copies) - 1:
                c1 = copies[i]
                c2 = copies[i+1]
                if (c1.src_buffer == c2.src_buffer and c1.dst_buffer == c2.dst_buffer and
                    c1.src_offset + c1.cnt == c2.src_offset and c1.dst_offset + c1.cnt == c2.dst_offset):
                    c1.cnt += c2.cnt
                    del copies[i+1]
                else:
                    i += 1
        combine_copies(gpu.precopies)
        combine_copies(gpu.postcopies)

    # Expand copies by instances if necessary
    if instances > 1:
        for rank, gpu in gpus.items():
            for copy in itertools.chain(gpu.precopies, gpu.postcopies):
                copy.src_offset *= instances
                copy.dst_offset *= instances
                copy.cnt *= instances

    def get_buffer_and_offset(gpu, addr):
        # Map an address to one of the named buffers
        if addr in gpu.inputs:
            return 'i', gpu.inputs[addr]
        elif addr in gpu.outputs:
            return 'o', gpu.outputs[addr]
        elif addr in gpu.scratch:
            return 's', gpu.scratch[addr]
        else:
            raise RuntimeError('Address is not mapped to a buffer')

    def categorize_ops(sends, initialized):
        sends_by_dest = defaultdict(set)
        for addr, src, dst in sends:
            dstbuf, dstoff = get_buffer_and_offset(gpus[dst], addr)
            sends_by_dest[(dst, dstbuf, dstoff)].add((src, addr))
        for key in sends_by_dest:
            dst, dstbuf, dstoff = key
            for idx, (src, addr) in enumerate(sends_by_dest[key]):
                # Receives into initialized buffer indices turn into reductions
                op_type = 'r' if idx == 0 and not (dstbuf, dstoff) in initialized[dst] else 'rrc'
                yield (addr, src, dst, op_type, idx)

    def make_intervals(src, dst, addrs_set):
        if len(addrs_set) == 0:
            return

        buffs_and_offs = []
        for addr, dst_op_type in addrs_set:
            srcbuff, srcoff = get_buffer_and_offset(gpus[src], addr)
            dstbuff, dstoff = get_buffer_and_offset(gpus[dst], addr)
            buffs_and_offs.append((srcbuff, srcoff, dstbuff, dstoff, dst_op_type))
        
        if merge_contiguous:
            # Sort sends by both buffers and offsets and merge sends into larger intervals when both the source and
            # destination are contiguous.
            buffs_and_offs.sort()
            start = prev = buffs_and_offs[0]

            def make_interval(a,b):
                cnt = b[1] - a[1] + 1
                assert cnt == b[3] - a[3] + 1, 'Source and destination count mismatch'
                return (a[0], a[1], a[2], a[3], a[4], cnt)
        
            for x in buffs_and_offs[1:]:
                if x[0] == prev[0] and x[1] == prev[1] + 1 and x[2] == prev[2] and x[3] == prev[3] + 1 and x[4] == prev[4]:
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
            for srcbuff, srcoff, dstbuff, dstoff, dst_op_type in buffs_and_offs:
                yield (srcbuff, srcoff, dstbuff, dstoff, dst_op_type, 1)    

    # Turn all steps of the algorithm into operations
    ops_by_channel = defaultdict(list)
    # Track the latest op that wrote to each buffer index
    writers = defaultdict(list)
    # Track all the reads since the last write to each buffer index
    readers = defaultdict(list)
    # Track which addresses are initialized on each rank
    initialized = [set(itertools.chain((('i', offset) for offset in gpu.inputs.values()),
        ((copy.dst_buffer, copy.dst_offset) for copy in gpu.precopies))) for gpu in gpus.values()]

    # Initialize readers and writers for precopies
    for rank, gpu in gpus.items():
        for op in gpu.precopies:
            for i in range(op.cnt):
                readers[(rank,op.src_buffer,op.src_offset+i)].append(op)
                writers[(rank,op.dst_buffer,op.dst_offset+i)].append(op)

    step_idx = 0
    for algo_step in algorithm.steps:
        # Categorize and serialize sends
        serialized_steps = []
        for addr, src, dst, dst_op_type, idx in categorize_ops(algo_step.sends, initialized):
            if idx >= len(serialized_steps):
                serialized_steps.extend([] for _ in range(idx - len(serialized_steps) + 1))
            serialized_steps[idx].append((addr, src, dst, dst_op_type))
        
        for categorized_sends in serialized_steps:
            new_writers = defaultdict(list)
            new_readers = defaultdict(list)

            # Group sent addresses by edge
            grouped_sends = defaultdict(set)
            for addr, src, dst, dst_op_type in categorized_sends:
                grouped_sends[(src,dst)].add((addr, dst_op_type))

            # Combine sends into intervals and create multiple instances if necessary
            sends = []
            for (src, dst), addrs in grouped_sends.items():
                intervals = list(make_intervals(src, dst, addrs))
                if channel_policy == ChannelPolicy.One:
                    num_chans = 1
                    channeled_intervals = [ (src_buf, src_off, dst_buf, dst_off, dst_op_type, cnt, 0) for src_buf, src_off, dst_buf, dst_off, dst_op_type, cnt in intervals ]
                elif channel_policy == ChannelPolicy.MatchTopology:
                    # Divide sends onto channels matching the topology (assume bw is ideal concurrency)
                    # Sends are split to balance channels if necessary
                    num_chans = algorithm.topology.link(src,dst)
                    channeled_intervals = []

                    intervals.sort(key=lambda x: x[-1])
                    counts = [x[-1] for x in intervals]
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
                        src_buf, src_off, dst_buf, dst_off, dst_op_type, cnt = intervals[i]
                        del intervals[i]
                        del counts[i]
                        if cnt > targets[chan]:
                            rem = cnt - targets[chan]
                            cnt = targets[chan]
                            j = bisect.bisect_left(counts, rem)
                            intervals.insert(j, (src_buf, src_off + cnt, dst_buf, dst_off + cnt, dst_op_type, rem))
                            counts.insert(j, rem)

                        channeled_intervals.append((src_buf, src_off, dst_buf, dst_off, dst_op_type, cnt, chan))
                        targets[chan] -= cnt
                        assert targets[chan] >= 0
                        if targets[chan] == 0:
                            chan += 1
                else:
                    assert False, 'Unhandled channel policy'

                for src_buf, src_off, dst_buf, dst_off, dst_op_type, cnt, chan in channeled_intervals:
                    for i in range(instances):
                        new_src_off = src_off * instances + i * cnt
                        new_dst_off = dst_off * instances + i * cnt
                        send = (src, dst, src_buf, new_src_off, dst_buf, new_dst_off, dst_op_type, cnt, chan * instances + i)
                        sends.append(send)

            # Perform dependency tracking and create _Op instances
            for src, dst, src_buf, src_off, dst_buf, dst_off, dst_op_type, cnt, chan in sends:
                read_keys = [(src,src_buf,src_off+i) for i in range(cnt)]
                # A send must wait for the previous recv (if any) to finish
                send_depends = list(set(d for k in read_keys for d in writers[k]))

                write_keys = [(dst,dst_buf,dst_off+i) for i in range(cnt)]
                # A receive must wait for both the previous recv and any previous sends to finish
                recv_depends = list(set(d for deps in (readers, writers) for k in write_keys for d in deps[k]))

                send_op = _Op(src, dst, step_idx, True, 's', src_buf, src_off, dst_buf, dst_off, cnt, send_depends)
                recv_op = _Op(dst, src, step_idx, False, dst_op_type, src_buf, src_off, dst_buf, dst_off, cnt, recv_depends)
                # Record the send and receive as a set of operations that must happen on the same channel
                ops_by_channel[chan].extend([send_op, recv_op])

                # Mark writers and readers to be added for the next step
                for k in write_keys:
                    new_writers[k].append(recv_op)
                for k in read_keys:
                    new_readers[k].append(send_op)
            # Writes cut the dependency to both previous writes and reads
            for key, deps in new_writers.items():
                if key in new_readers:
                    gpu, buf, off = key
                    raise RuntimeError(f'Encountered receive and send on the same buffer index on step {step_idx + 1} (gpu={gpu}, buf={buf}, off={off})')
                writers[key] = deps
                readers[key] = []
            # Reads get added to any previous reads
            for key, deps in new_readers.items():
                readers[key].extend(deps)
            # Update initialized sets
            for ops in ops_by_channel.values():
                for op in ops:
                    if not op.is_send:
                        initialized[op.gpu].add((op.dst_buffer, op.dst_offset))
            step_idx += 1

    # Add dependencies for postcopies
    for rank, gpu in gpus.items():
        for op in gpu.postcopies:
            for i in range(op.cnt):
                op.depends.extend(writers[(rank,op.src_buffer,op.src_offset+i)])
                op.depends.extend(readers[(rank,op.dst_buffer,op.dst_offset+i)])
                op.depends.extend(writers[(rank,op.dst_buffer,op.dst_offset+i)])

    # Fixup everything to match the instanced sends when multiple instances are generated
    if instances > 1:
        for rank, gpu in gpus.items():
            # Multiply metadata with instances
            def expand_mappings(mappings):
                return { addr * instances + i: idx * instances + i for addr, idx in mappings.items() for i in range(instances) }
            gpu.inputs = expand_mappings(gpu.inputs)
            gpu.outputs = expand_mappings(gpu.outputs)
            gpu.input_chunks *= instances
            gpu.output_chunks *= instances
            gpu.scratch = expand_mappings(gpu.scratch)

    # Group by which operations need to be in the same threadblock
    tb_groups = defaultdict(list)
    for chan, chan_ops in ops_by_channel.items():
        for op in chan_ops:
            tb_groups[(op.gpu, op.is_send, op.peer, chan)].append(op)

    tbs_by_gpu_chan = defaultdict(lambda: defaultdict(list))
    # For each group find or create a threadblock to add them to
    for key, grp in tb_groups.items():
        rank, is_send, peer, chan = key
        tbs = tbs_by_gpu_chan[rank][chan]
        for tb in tbs:
            tb_peer = tb.send if is_send else tb.recv
            # An existing threadblock can be reused if:
            # - Either the relevant peer is not set yet or the peer is the same
            # - No operations already in the threadblock execute in the same step
            if tb_peer == -1 or tb_peer == peer:
                if all(not any(op1.step == op2.step for op2 in grp) for op1 in tb.steps):
                    break
        else:
            # No existing threadblock was suitble, so create a new one
            tb = _Threadblock(chan)
            tbs.append(tb)
        # Ensure the peer is set correctly
        if is_send:
            assert tb.send == -1 or tb.send == peer
            tb.send = peer
        else:
            assert tb.recv == -1 or tb.recv == peer
            tb.recv = peer
        tb.steps.extend(grp)

    # Sort threadblocks in each GPU by peers and then the channel
    # This is important as in NCCL threadblocks using the same NVLink concurrently should be close together
    for rank, gpu in gpus.items():
        gpu.threadblocks = sorted([tb for tbs in tbs_by_gpu_chan[rank].values() for tb in tbs],
            key=lambda tb: (tb.send, tb.recv, tb.channel))
        for i, tb in enumerate(gpu.threadblocks):
            tb.rbid = i

    # Add all copies into extra threadblocks
    for rank, gpu in gpus.items():
        cpy_tb = _Threadblock(0)
        cpy_tb.rbid = len(gpu.threadblocks)
        cpy_tb.steps = gpu.precopies + gpu.postcopies
        gpu.threadblocks.append(cpy_tb)

    # Filter out dependencies within the same threadblock and mark all ops that have a dependence on them
    for rank, gpu in gpus.items():
        for tb in gpu.threadblocks:
            for op in tb.steps:
                op.block_rbid = tb.rbid
    for rank, gpu in gpus.items():
        for tb in gpu.threadblocks:
            for op in tb.steps:
                op.depends = list(filter(lambda d: d.block_rbid != op.block_rbid, op.depends))
                for dep in op.depends:
                    dep.has_dependence = True

    # Do some additional postprocessing of operations:
    # - Expand operations with extra dependencies with no-ops
    # - Mark the index of each operation taking any extra no-ops into account
    for rank, gpu in gpus.items():
        for tb in gpu.threadblocks:
            tb.steps.sort(key=lambda op: op.step)
            for op in tb.steps:
                # Expand extra dependencies into nop operations
                if len(op.depends) > 1:
                    extra_deps = op.depends[1:]
                    op.depends = op.depends[:1]
                    first_step = op.step
                    for i, dep in enumerate(extra_deps):
                        tb.ops.append(_Op(op.gpu, None, op.step, False, 'nop', None, None, None, None, 0, [dep]))
                        tb.ops[-1].idx = len(tb.ops) - 1
                tb.ops.append(op)
                tb.ops[-1].idx = len(tb.ops) - 1

    # Generate the XML structure
    algo_elem = ET.Element('algo')
    algo_elem.set('name', algorithm.name)
    algo_elem.set('proto', 'Simple')
    nchannels = 1 + max(max(tb.channel for tb in gpu.threadblocks) for gpu in gpus.values())
    algorithm.nchannels = nchannels
    algo_elem.set('nchannels', str(nchannels))
    algo_elem.set('ngpus', str(len(gpus)))
    algo_elem.set('inplace', '0')
    algo_elem.set('coll', algorithm.collective.runtime_name)
    algo_elem.set('nchunksperloop', str(max(max(gpu.input_chunks, gpu.output_chunks) for gpu in gpus.values())))
    for rank, gpu in gpus.items():
        gpu_elem = ET.SubElement(algo_elem, 'gpu')
        gpu_elem.set('id', str(rank))
        gpu_elem.set('i_chunks', str(gpu.input_chunks))
        gpu_elem.set('o_chunks', str(gpu.output_chunks))
        gpu_elem.set('s_chunks', str(gpu.scratch_size()))
        for tb in gpu.threadblocks:
            tb_elem = ET.SubElement(gpu_elem, 'tb')
            tb_elem.set('id', str(tb.rbid))
            tb_elem.set('send', str(tb.send))
            tb_elem.set('recv', str(tb.recv))
            tb_elem.set('chan', str(tb.channel))
            for op in tb.ops:
                op_elem = ET.SubElement(tb_elem, 'step')
                op_elem.set('s', str(op.idx))
                op_elem.set('type', op.op_type)

                # The NCCL backend currently wants scratch at the end of output
                if not use_scratch:
                    if op.src_buffer == 's':
                        op.src_buffer = 'o'
                        op.src_offset += gpu.output_chunks
                    if op.dst_buffer == 's':
                        op.dst_buffer = 'o'
                        op.dst_offset += gpu.output_chunks

                if op.src_buffer is not None:
                    op_elem.set('srcbuf', op.src_buffer)
                    op_elem.set('srcoff', str(op.src_offset))
                else:
                    op_elem.set('srcbuf', 'i')
                    op_elem.set('srcoff', '-1')
                if op.dst_buffer is not None:
                    op_elem.set('dstbuf', op.dst_buffer)
                    op_elem.set('dstoff', str(op.dst_offset))
                else:
                    op_elem.set('dstbuf', 'o')
                    op_elem.set('dstoff', '-1')
                op_elem.set('cnt', str(op.cnt))
                assert len(op.depends) <= 1
                if len(op.depends) == 1:
                    op_elem.set('depid', str(op.depends[0].block_rbid))
                    op_elem.set('deps', str(op.depends[0].idx))
                else:
                    op_elem.set('depid', '-1')
                    op_elem.set('deps', '-1')
                if op.has_dependence:
                    op_elem.set('hasdep', '1')
                else:
                    op_elem.set('hasdep', '0')

    if pretty_print:
        ET.indent(algo_elem, space='  ')
    return ET.tostring(algo_elem, encoding='unicode')
