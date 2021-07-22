# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from lxml import etree as ET
from collections import defaultdict
from dataclasses import dataclass, field, replace
import math
import threading, queue
from enum import Enum
from z3 import *

@dataclass
class _Gpu:
    copies: list
    inputs: dict
    outputs: dict
    input_chunks: int
    output_chunks: int
    scratch: dict = field(default_factory=dict)
    threadbloks: list = field(default_factory=list)

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
class _Copy:
    input_offset: int
    output_offset: int

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

    # For copies reserve the index in the output buffer from the very beginning
    for rank, gpu in gpus.items():
        for copy in gpu.copies:
            output_livenesses[rank][copy.output_offset] = [(-1,math.inf)]

    def update_liveness(rank, addr, step_idx):
        gpu = gpus[rank]
        # Find the relevant buffer and livenesses for the address
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

def _allocate_channels_max_concurrency(op_sets, logging):
    # This function solves a coloring problem to ops to a minimal set of channels
    ctx = Context()

    def chan(idx):
        return Int(f'chan_{idx}', ctx=ctx)
    max_channels = Int('max_channels', ctx=ctx)

    constraints = []

    # Add basic constraints and find conflicting sets of operations
    conflict_groups = defaultdict(set)
    for idx, op_set in enumerate(op_sets):
        for op in op_set:
            # Two operations conflict if they use the same src-dst edge on the same step
            conflict_groups[(op.gpu, op.is_send, op.peer, op.step)].add(idx)
        constraints.append(chan(idx) >= 0)
        constraints.append(chan(idx) < max_channels)

    # Require channels within the conflict groups to be disjoint
    for grp in conflict_groups.values():
        constraints.append(Distinct([chan(idx) for idx in grp]))

    opt = Optimize(ctx=ctx)
    opt.add(constraints)
    opt.minimize(max_channels)
    
    t = threading.Thread(target=opt.check)
    t.start()
    t.join(1)
    main_ctx().interrupt()
    t.join()

    try:
        model = opt.model()
    except Z3Exception:
        # TODO: This altenate process does not guarantee that channels are contiguous
        s = Solver(ctx=ctx)
        s.add(constraints)
        s.check()
        model = s.model()
            
    if logging:
        print(f'Using up to {model[max_channels].as_long()} channels')

    # Group the operations by which channels they use
    ops_by_channel = defaultdict(list)
    for idx, op_set in enumerate(op_sets):
        ops = ops_by_channel[model[chan(idx)].as_long()]
        ops.extend(op_set)

    return ops_by_channel

def _allocate_channels_match_topology(op_sets, topology, instances, logging):
    if len(topology.switches) > 0 and logging:
        print('Warning: Switches in the topology are ignored for the channel policy MatchTopology.')

    ops_by_channel = defaultdict(list)
    next_channel = defaultdict(lambda: 0)
    for op_set in op_sets:
        send = op_set[0]
        assert send.op_type == 's'
        src = send.gpu
        dst = send.peer
        ops_by_channel[next_channel[(src,dst)]].extend(op_set)
        link = topology.link(src,dst) * instances
        assert link > 0, 'Encountered send on non-existent link'
        next_channel[(src,dst)] = (next_channel[(src,dst)] + 1) % link

    return ops_by_channel

class ChannelPolicy(Enum):
    One = 'One'
    MaxConcurrency = 'MaxConcurrency'
    MatchTopology = 'MatchTopology'

    def __str__(self):
        return self.value

def ncclize(algorithm, remap_scratch = None, channel_policy=ChannelPolicy.MatchTopology, pretty_print = True, old_format=False, use_scratch=False, merge_contiguous=True, instances=1, logging=False):
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
        copies = []
        if rank in algorithm.input_map:
            for idx, addr in enumerate(sorted(algorithm.input_map[rank])):
                if addr in outputs:
                    copies.append(_Copy(idx, outputs[addr]))
                else:
                    inputs[addr] = idx
        gpus[rank] = _Gpu(copies, inputs, outputs, len(inputs) + len(copies), len(outputs))

    # Create scratch buffer mappings if necessary
    def allocate_scratch(gpu, addr):
        if not (addr in gpu.inputs or addr in gpu.outputs or addr in gpu.scratch):
            offset = len(gpu.scratch)
            gpu.scratch[addr] = offset
    for step in algorithm.steps:
        for addr, src, dst in step.sends:
            allocate_scratch(gpus[src], addr)
            allocate_scratch(gpus[dst], addr)

    # Analyze liveness of indices in buffers and remap scratch into input/output as possible
    if remap_scratch:
        liveness = _analyze_liveness(gpus, algorithm)
        _remap_scratch_into_input_output(liveness, gpus, logging)

    # Sort scratch mappings in an attemp to make more of them contiguous (this is of course a heuristic).
    for gpu in gpus.values():
        gpu.scratch = { addr: idx for idx, addr in enumerate(sorted(gpu.scratch)) }

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
    op_sets = []
    # Track the latest op that wrote to each buffer index
    writers = defaultdict(list)
    # Track all the reads since the last write to each buffer index
    readers = defaultdict(list)
    for step_idx, step in enumerate(algorithm.steps):
        new_writers = defaultdict(list)
        new_readers = defaultdict(list)

        # Group sent addresses by edge
        grouped_sends = defaultdict(set)
        for addr, src, dst in step.sends:
            grouped_sends[(src,dst)].add(addr)

        # Combine sends into intervals and create multiple instances if necessary
        sends = []
        for (src, dst), addrs in grouped_sends.items():
            for src_buf, src_off, dst_buf, dst_off, cnt in make_intervals(src, dst, addrs):
                for i in range(instances):
                    new_src_off = src_off * instances + i * cnt
                    new_dst_off = dst_off * instances + i * cnt
                    send = (src, dst, src_buf, new_src_off, dst_buf, new_dst_off, cnt)
                    sends.append(send)

        # Perform dependency tracking and create _Op instances
        for src, dst, src_buf, src_off, dst_buf, dst_off, cnt in sends:
            read_keys = [(src,src_buf,src_off+i) for i in range(cnt)]
            # A send must wait for the previous recv (if any) to finish
            send_depends = list(set(d for k in read_keys for d in writers[k]))

            write_keys = [(dst,dst_buf,dst_off+i) for i in range(cnt)]
            # A receive must wait for both the previous recv and any previous sends to finish
            recv_depends = list(set(d for deps in (readers, writers) for k in write_keys for d in deps[k]))

            send_op = _Op(src, dst, step_idx, True, 's', src_buf, src_off, dst_buf, dst_off, cnt, send_depends)
            recv_op = _Op(dst, src, step_idx, False, 'r', src_buf, src_off, dst_buf, dst_off, cnt, recv_depends)
            # Record the send and receive as a set of operations that must happen on the same channel
            op_sets.append([send_op, recv_op])

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

    # Fixup everything to match the instanced sends when multiple instances are generated
    if instances > 1:
        for gpu in gpus.values():
            # Create instances copies of the copies.
            new_copies = []
            for copy in gpu.copies:
                for i in range(instances):
                    new_copy = _Copy(copy.input_offset * instances + i, copy.output_offset * instances + i)
                    new_copies.append(new_copy)
            gpu.copies = new_copies

            # Multiply the other metadata with instances
            def expand_mappings(mappings):
                return { addr * instances + i: idx * instances + i for addr, idx in mappings.items() for i in range(instances) }
            gpu.inputs = expand_mappings(gpu.inputs)
            gpu.outputs = expand_mappings(gpu.outputs)
            gpu.input_chunks *= instances
            gpu.output_chunks *= instances
            gpu.scratch = expand_mappings(gpu.scratch)

    # Allocate channels and group operations by channel
    if channel_policy == ChannelPolicy.One:
        ops_by_channel = {0: [op for op_set in op_sets for op in op_set]}
    elif channel_policy == ChannelPolicy.MaxConcurrency:
        ops_by_channel = _allocate_channels_max_concurrency(op_sets, logging)
    elif channel_policy == ChannelPolicy.MatchTopology:
        ops_by_channel = _allocate_channels_match_topology(op_sets, algorithm.topology, instances, logging)
    else:
        assert False, 'Unhandled channel policy'

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

    # Do some additional postprocessing of operations:
    # - Expand operations with extra dependencies with no-ops
    # - Mark the index of each operation taking any extra no-ops into account
    # - Record the threadblock rbids for each operation
    all_ops = []
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
            for op in tb.ops:
                op.block_rbid = tb.rbid
            all_ops.extend(tb.ops)

    # Filter out dependencies within the same threadblock
    for op in all_ops:
        op.depends = list(filter(lambda d: d.block_rbid != op.block_rbid, op.depends))

    # Mark all ops that have a dependence on them
    for op in all_ops:
        for dep in op.depends:
            dep.has_dependence = True

    # Generate the XML structure
    algo_elem = ET.Element('algo')
    algo_elem.set('name', algorithm.name)
    algo_elem.set('proto', 'Simple')
    nchannels = 1 + max(max(tb.channel for tb in gpu.threadblocks) for gpu in gpus.values())
    algorithm.nchannels = nchannels
    algo_elem.set('nchannels', str(nchannels))
    if old_format:
        algo_elem.set('nchunksperloop', str(max(max(gpu.input_chunks, gpu.output_chunks) for gpu in gpus.values())))
    for rank, gpu in gpus.items():
        gpu_elem = ET.SubElement(algo_elem, 'gpu')
        gpu_elem.set('id', str(rank))
        gpu_elem.set('i_chunks', str(gpu.input_chunks))
        gpu_elem.set('o_chunks', str(gpu.output_chunks))
        gpu_elem.set('s_chunks', str(gpu.scratch_size()))
        for copy in gpu.copies:
            copy_elem = ET.SubElement(gpu_elem, 'copy')
            copy_elem.set('i_off', str(copy.input_offset))
            copy_elem.set('o_off', str(copy.output_offset))
        for tb in gpu.threadblocks:
            tb_elem = ET.SubElement(gpu_elem, 'tb')
            tb_elem.set('id', str(tb.rbid))
            tb_elem.set('send', str(tb.send))
            tb_elem.set('recv', str(tb.recv))
            tb_elem.set('chan', str(tb.channel))
            for op in tb.ops:
                op_elem = ET.SubElement(tb_elem, 'op' if not old_format else 'step')
                op_elem.set('step' if not old_format else 's', str(op.idx))
                op_elem.set('type', op.op_type)

                # The NCCL backend currently wants scratch at the end of output
                if not use_scratch:
                    if op.src_buffer == 's':
                        op.src_buffer = 'o'
                        op.src_offset += gpu.output_chunks
                    if op.dst_buffer == 's':
                        op.dst_buffer = 'o'
                        op.dst_offset += gpu.output_chunks

                if old_format:
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
                else:
                    if op.is_send:
                        if op.src_buffer is not None:
                            op_elem.set('buf', op.src_buffer)
                            op_elem.set('off', str(op.src_offset))
                    else:
                        if op.dst_buffer is not None:
                            op_elem.set('buf', op.dst_buffer)
                            op_elem.set('off', str(op.dst_offset))
                if op.cnt > 1 or old_format:
                    op_elem.set('cnt', str(op.cnt))
                assert len(op.depends) <= 1
                if len(op.depends) == 1:
                    op_elem.set('depid', str(op.depends[0].block_rbid))
                    op_elem.set('deps', str(op.depends[0].idx))
                elif old_format:
                    op_elem.set('depid', '-1')
                    op_elem.set('deps', '-1')
                if op.has_dependence:
                    op_elem.set('hasdep', '1')
                elif old_format:
                    op_elem.set('hasdep', '0')

    if pretty_print:
        ET.indent(algo_elem, space='  ')
    return ET.tostring(algo_elem, encoding='unicode')
