# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from enum import Enum
import heapq
import functools

from msccl.language.ir import *
from msccl.language.passes import *

def remove_op(op):
    for p in op.prev:
        p.next.remove(op)
        p.next += op.next

    for n in op.next:
        n.prev.remove(op)
        n.prev = op.prev.union(n.prev)

def merge_op(op, other_op):
    for p in other_op.prev:
        p.next.remove(other_op)
        p.next.append(op)

    for n in other_op.next:
        n.prev.remove(other_op)
        n.prev.add(op)

    op.prev = op.prev.union(other_op.prev)
    op.next += other_op.next

def same_tb(op1, op2):
    return op1.tb == op2.tb and op1.channel == op2.channel

def same_count(op1, op2):
    return op1.cnt() == op2.cnt()

def same_buf_dst(op1, op2):
    return op1.dst.buffer == op2.dst.buffer and op1.dst.index == op2.dst.index

def same_src_dst_buffer_type(op1, op2):
    return op1.src.buffer == op2.src.buffer and op1.dst.buffer == op2.dst.buffer

def buf_dst_src_match(op1, op2):
    return op1.dst.buffer == op2.src.buffer and op1.dst.index == op2.src.index

def same_buf_src(op1, op2):
    return op1.src.buffer == op2.src.buffer and op1.src.index == op2.src.index

def same_chan_type(op1, op2):
    return op1.channel_type == op2.channel_type

class InstructionDAG:
    def __init__(self, num_ranks, buffers):
        self.num_ranks = num_ranks
        self.buffers = buffers
        # State for the actual instruction DAG
        self.operations = {} # slot -> operations
        self.last_writer = {} # slot -> last writing op
        self.last_readers = defaultdict(list) # slot -> list of last reading ops
        # State for the MSCCL-IR
        self.tbs = []
        for _ in range(num_ranks):
            self.tbs.append({})
        self.tb_mapping = {}
        self.num_channels = [1] * num_ranks
        self.tb_steps = [{} for _ in range(num_ranks)]

    # InstructionDAG helper - identifies the dependencies for a write-type operation (recv, copy, rrc, reduce)
    def _write(self, rank, buffer, index, size, op, read=False):
        prev_ops = set()
        for i in range(index, index+size):
            slot = (rank, buffer, i)
            if read:
                assert slot in self.last_writer, f"Destination slot has never been written before a reduce {op}"

            # First write to this slot
            if slot not in self.operations:
                self.operations[slot] = op

            # If there are active readers - these are the previous operations
            # Else the previous operation is the last write (if there is one)
            readers = self.last_readers[slot]
            if len(readers) > 0:
                prev_ops.update(readers)
            elif slot in self.last_writer:
                prev_ops.add(self.last_writer[slot])

            # Set the last_writer to this op, and clear all readers
            self.last_writer[slot] = op
            self.last_readers[slot] = []

        # Update the next pointer of the previous ops
        for prev_op in prev_ops:
            prev_op.next.add(op)
            op.prev.add(prev_op)

    # InstructionDAG helper - identifies the dependencies for read-type operations (send, copy, reduce)
    def _read(self, rank, buffer, index, size, op):
        prev_ops = set()
        for i in range(index, index+size):
            slot = (rank, buffer, i)
            assert slot in self.last_writer, f"Slot has never been written before a read-type {op}"
            # The previous operation for a reader is the last write to the slot
            writer = self.last_writer[slot]
            prev_ops.add(writer)
            self.last_readers[slot].append(op)

        # Update the next pointer of the previous ops
        for prev_op in prev_ops:
            prev_op.next.add(op)
            op.prev.add(prev_op)

    # InstructionDAG - builds the roots of the DAG
    def add_start(self, rank, buffer, index, ref):
        slot = (rank, buffer, index)
        op = Op(Instruction.start, rank, ref, ref, next=set(), prev=set(), chunk_step=-1)
        self.operations[slot] = op
        self.last_writer[slot] = op

    # InstructionDAG - adds a copy node
    def add_copy(self, rank, send_ref, recv_ref, tb, ch):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(Instruction.copy, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel=ch, step=tb_step)
        dstbuffer = recv_ref.buffer
        dstindex = recv_ref.index
        srcbuffer = send_ref.buffer
        srcindex = send_ref.index
        size = recv_ref.size
        # Sending part of copy [Read]
        self._read(rank, srcbuffer, srcindex, size, op)
        # Receiving part of copy [Write]
        self._write(rank, dstbuffer, dstindex, size, op)
        return op

    # InstructionDAG - adds a redduce node
    def add_reduce(self, rank, send_ref, recv_ref, tb, ch):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(Instruction.reduce, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel=ch, step=tb_step)
        dstbuffer = recv_ref.buffer
        dstindex = recv_ref.index
        srcbuffer = send_ref.buffer
        srcindex = send_ref.index
        size = recv_ref.size
        prev_ops = []
        op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        # Sending part of reduce
        self._read(rank, srcbuffer, srcindex, size, op)
        # Reduce part of copy
        self._write(rank, dstbuffer, dstindex, size, op, read=True)
        return op

    # InstructionDAG - adds a send node
    def add_send(self, rank, send_ref, recv_ref, tb, ch):
        op = Op(Instruction.send, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel=ch)
        buffer = send_ref.buffer
        index = send_ref.index
        size = send_ref.size
        self._read(rank, buffer, index, size, op)
        return op

    # InstructionDAG - adds a put node
    def add_put(self, rank, send_ref, recv_ref, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(Instruction.put, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel_type=ch_type, step=tb_step)
        buffer = send_ref.buffer
        index = send_ref.index
        size = send_ref.size
        self._read(rank, buffer, index, size, op)
        return op

    def add_get(self, rank, send_ref, recv_ref, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(Instruction.get, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel_type=ch_type, step=tb_step)
        buffer = recv_ref.buffer
        index = recv_ref.index
        size = recv_ref.size
        self._write(rank, buffer, index, size, op)
        return op

    # InstructionDAG - adds a signal node.
    def add_signal(self, rank, send_ref, recv_ref, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(Instruction.signal, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel_type=ch_type, step=tb_step)
        buffer = send_ref.buffer
        index = send_ref.index
        size = send_ref.size
        # treat signal as a write since it can not be executed parallelly with read operations
        self._write(rank, buffer, index, size, op)
        op.dsts.append((ChunkRef(recv_ref.rank, recv_ref.buffer, recv_ref.index, recv_ref.size), tb_step))
        op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        return op

    def add_wait(self, rank, dst_ref, src_ref, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(Instruction.wait, rank, src_ref, dst_ref, next=set(), prev=set(), tb=tb, channel_type=ch_type, step=tb_step)
        buffer = dst_ref.buffer
        index = dst_ref.index
        size = dst_ref.size
        self._write(rank, buffer, index, size, op)
        op.srcs.append((ChunkRef(src_ref.rank, src_ref.buffer, src_ref.index, src_ref.size), tb_step))
        op.dsts.append((ChunkRef(dst_ref.rank, dst_ref.buffer, dst_ref.index, dst_ref.size), tb_step))
        return op

    # InstructionDAG - adds a recv node
    def add_recv(self, rank, send_ref, recv_ref, tb, ch, send_op):
        op = Op(Instruction.recv, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel=ch)
        buffer = recv_ref.buffer
        index = recv_ref.index
        size = recv_ref.size
        self._write(rank, buffer, index, size, op)
        op.send_match = send_op
        return op

    # InstructionDAG - adds a rrc node
    def add_recv_reduce_copy(self, rank, send_ref, recv_ref, tb, ch, send_op):
        op = Op(Instruction.recv_reduce_copy, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel=ch)
        buffer = recv_ref.buffer
        index = recv_ref.index
        size = recv_ref.size
        self._write(rank, buffer, index, size, op, read=True)
        op.send_match = send_op
        return op

    def add_read_reduce_copy(self, rank, send_ref, recv_ref, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(Instruction.read_reduce_copy, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel_type=ch_type, step=tb_step)
        buffer = recv_ref.buffer
        index = recv_ref.index
        size = recv_ref.size
        op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        self._write(rank, buffer, index, size, op, read=True)
        return op

    def convert_set_list(self):
        ops = []
        visited = set()
        for slot, op in self.operations.items():
            if op.inst == Instruction.start:
                op.next = list(op.next)
                for o in op.next:
                    ops.append(o)
            elif op.inst != Instruction.copy:
                ops.append(op)

            while len(ops) > 0:
                op = ops[0]
                if op not in visited:
                    visited.add(op)
                    op.next = list(op.next)
                    ops = ops[1:] + op.next
                else:
                    ops = ops[1:]
        return visited

    def optimize(self):
        self._optimize_rrcs_rrs()
        self._optimize_rcs()

    def complete_channels(self):
        send_op = [Instruction.put, Instruction.signal]
        recv_op = [Instruction.wait, Instruction.get, Instruction.read_reduce_copy]
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                chans = set()
                for op in tb.ops:
                    if op.inst in send_op:
                        chan = Channel(op.src.buffer, op.dst.buffer, op.channel_type, op.dst.rank)
                        chans.add(chan)
                    elif op.inst in recv_op:
                        chan = Channel(op.dst.buffer, op.src.buffer, op.channel_type, op.src.rank)
                        chans.add(chan)
                tb.channels = list(chans)

    def _optimize_redandant_signal_wait(self, protocol):
        if protocol != 'LL':
            return
        # For LL algorithm, we can remove signal/wait
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    if op.inst == Instruction.put:
                        fused = False
                        for next_op in op.next:
                            if next_op.inst == Instruction.signal:
                                remove_op(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    elif op.inst == Instruction.reduce or op.inst == Instruction.read_reduce_copy or op.inst == Instruction.copy:
                        fused = False
                        for prev_op in op.prev:
                            if prev_op.inst == Instruction.wait:
                                remove_op(prev_op)
                                fused = True
                                break
                        if fused:
                            continue
                    queue = queue[1:]

    # rrc(_,_,_,dst,dbuf,di) rrc(_,_,_,dst,dbuf,di) -> rrc(list[src,sbuf,si], dst, dbuf, di)
    # signal(_,_,_,dst,dbuf,di) signal(_,_,_,dst,dbuf,di) -> signal(_,_,_,list[dst,dbuf,di])
    # wait(src,sbuf,si,_,_,_) wait(src,sbuf,si,_,_,_) -> wait(list[src,sbuf,si],_,_,_,_])
    # reduce(_,_,_,dst,dbuf,di) reduce(_,_,_,dst,dbuf,di) -> reduce(list[src,sbuf,si], dst, dbuf, di)
    def _optimize_rrc_r_signal_wait(self):
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    if op.inst == Instruction.read_reduce_copy:
                        fused = False
                        for next_op in op.next:
                            if next_op.inst == Instruction.read_reduce_copy and same_count(op, next_op) and same_buf_dst(op, next_op) and same_chan_type(op, next_op):
                                op.srcs.append((ChunkRef(next_op.src.rank, next_op.src.buffer, next_op.src.index, next_op.src.size), next_op.step))
                                remove_op(next_op)
                                tb.ops.remove(next_op)
                                queue.remove(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    elif op.inst == Instruction.reduce:
                        fused = False
                        for next_op in op.next:
                            if next_op.inst == Instruction.reduce and same_buf_dst(op, next_op) and same_chan_type(op, next_op):
                                op.srcs.append((ChunkRef(next_op.src.rank, next_op.src.buffer, next_op.src.index, next_op.src.size), next_op.step))
                                remove_op(next_op)
                                tb.ops.remove(next_op)
                                queue.remove(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    elif op.inst == Instruction.signal:
                        fused = False
                        for next_op in op.next:
                            if next_op.inst == Instruction.signal and same_buf_src(op, next_op) and same_chan_type(op, next_op):
                                op.dsts.append((ChunkRef(next_op.dst.rank, next_op.dst.buffer, next_op.dst.index, next_op.dst.size), next_op.step))
                                op.srcs.append((ChunkRef(next_op.src.rank, next_op.src.buffer, next_op.src.index, next_op.src.size), next_op.step))
                                remove_op(next_op)
                                tb.ops.remove(next_op)
                                queue.remove(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    elif op.inst == Instruction.wait:
                        fused = False
                        for next_op in op.next:
                            if next_op.inst == Instruction.wait and same_buf_dst(op, next_op) and same_chan_type(op, next_op):
                                op.srcs.append((ChunkRef(next_op.src.rank, next_op.src.buffer, next_op.src.index, next_op.src.size), next_op.step))
                                op.dsts.append((ChunkRef(next_op.dst.rank, next_op.dst.buffer, next_op.dst.index, next_op.dst.size),next_op.step))
                                remove_op(next_op)
                                tb.ops.remove(next_op)
                                queue.remove(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    queue = queue[1:]

    # rrc(_,_,_,dst,dbuf,di) put(dst,dbuf,di,_,_,_) -> rrcs(_,_,_,_,_,_)
    # reduce(_,_,_,dst,dbuf,di) put(dst,dbuf,di,_,_,_) -> rs(_,_,_,_,_,_)
    def _optimize_rrcs_rs(self):
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    if op.inst == Instruction.read_reduce_copy or op.inst == Instruction.read_reduce_copy_send:
                        fused = False
                        for next_op in op.next:
                            if next_op.inst == Instruction.put and same_count(op, next_op) and buf_dst_src_match(op, next_op) and same_chan_type(op, next_op):
                                if len(op.dsts) > 0 and op.dsts[0][0].buffer != next_op.dst.buffer:
                                    continue
                                if op.inst == Instruction.read_reduce_copy:
                                    op.inst = Instruction.read_reduce_copy_send
                                op.dsts.append((ChunkRef(next_op.dst.rank, next_op.dst.buffer, next_op.dst.index, next_op.dst.size), next_op.step))
                                remove_op(next_op)
                                tb.ops.remove(next_op)
                                queue.remove(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    if op.inst == Instruction.reduce or op.inst == Instruction.reduce_send:
                        fused = False
                        for next_op in op.next:
                            if next_op.inst == Instruction.put and same_count(op, next_op) and buf_dst_src_match(op, next_op) and next_op.channel_type == ChannelType.sm:
                                if len(op.dsts) > 0 and op.dsts[0][0].buffer != next_op.dst.buffer:
                                    continue
                                if op.inst == Instruction.reduce:
                                    op.inst = Instruction.reduce_send
                                op.dsts.append((ChunkRef(next_op.dst.rank, next_op.dst.buffer, next_op.dst.index, next_op.dst.size), next_op.step))
                                remove_op(next_op)
                                tb.ops.remove(next_op)
                                queue.remove(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    queue = queue[1:]

    # For signal/wait ops, if they are independent of other operations and no other operations in between,
    # then merge them into a single signal/wait op
    # wait(src,sbuf,si,_,_,_) wait(src,sbuf,si,_,_,_) -> wait(list[src,sbuf,si],_,_,_,_])
    def _parallel_signal_wait(self):
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                if tbid == -1:
                    continue
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    if op.inst == Instruction.signal:
                        fused = False
                        if len(queue) > 1:
                            seq_op = queue[1]
                            if seq_op.inst == Instruction.signal and same_src_dst_buffer_type(op, seq_op) and same_chan_type(op, seq_op):
                                op.dsts.append((ChunkRef(seq_op.dst.rank, seq_op.dst.buffer, seq_op.dst.index, seq_op.dst.size), seq_op.step))
                                op.srcs.append((ChunkRef(seq_op.src.rank, seq_op.src.buffer, seq_op.src.index, seq_op.src.size), seq_op.step))
                                merge_op(op, seq_op)
                                tb.ops.remove(seq_op)
                                queue.remove(seq_op)
                                fused = True
                        if fused:
                            continue
                    elif op.inst == Instruction.wait:
                        fused = False
                        if len(queue) > 1:
                            seq_op = queue[1]
                            if seq_op.inst == Instruction.wait and same_src_dst_buffer_type(op, seq_op) and same_chan_type(op, seq_op):
                                op.dsts.append((ChunkRef(seq_op.dst.rank, seq_op.dst.buffer, seq_op.dst.index, seq_op.dst.size), seq_op.step))
                                op.srcs.append((ChunkRef(seq_op.src.rank, seq_op.src.buffer, seq_op.src.index, seq_op.src.size), seq_op.step))
                                merge_op(op, seq_op)
                                tb.ops.remove(seq_op)
                                queue.remove(seq_op)
                                fused = True
                        if fused:
                            continue
                    queue = queue[1:]

    def optimize_mscclpp(self, protocol):
        self._optimize_redandant_signal_wait(protocol)
        self._optimize_rrc_r_signal_wait()
        self._optimize_rrcs_rs()

        self._parallel_signal_wait()

    # Completes metadata for chunk_steps (number of steps from a start op) and priority (number of steps to the last op)
    def _complete_metadata(self):
        def dfs(op, cs):
            op.chunk_step = max(op.chunk_step, cs+1)

            if len(op.next) == 0 and op.recv_match is None:
                op.priority = 0
            else:
                for o in op.next:
                    dfs(o, op.chunk_step)
                # Priority = +1 of the highest priority child
                if len(op.next) > 0:
                    highest_next_priority = max([x.priority+1 for x in op.next])
                    op.priority = max(highest_next_priority, op.priority)
                if op.is_send():
                    dfs(op.recv_match, op.chunk_step)
                    op.priority = max(op.priority, op.recv_match.priority+1)

        for chunk, op in self.operations.items():
            if op.inst == Instruction.start:
                dfs(op,-2) # Start instructions should start at -1


    # Given the set of operations that operate over a particular slot (rank, buffer, idx) fixed
    # Try and replace operations with pipelined ops like receive copy send (rcs)
    # or receive reduce send (rrs) and receive reduce copy send (rrcs)
    # Rules:
    # recv-copy-send
    # recv(src, sbuf, si, _, _, _ ) send(_, _, _, dst, dbuf, di) -> recv_copy_send(src, sbuf, si, dst, dbuf, di)
    def _optimize_rcs(self):
        for slot, ops in self.operations.items():
            frontier = [ops]
            while len(frontier) > 0:
                op = frontier[0]
                for next_op in op.next:
                    if op.inst == Instruction.recv and next_op.inst == Instruction.send and same_tb(op, next_op) and same_count(op, next_op) and same_buf_dst(op, next_op):
                        # recv -> rcs, remove send
                        op.inst = Instruction.recv_copy_send
                        op.dst = next_op.dst
                        next_op.recv_match.send_match = op
                        op.recv_match = next_op.recv_match
                        remove_op(next_op)
                        break
                frontier = frontier[1:] + op.next
    # recv-reduce-send - A rrc followed by a send that gets overwritten
    # rrc(src, sbuf, si, ...) send(_, _, _, dst, dbuf, di) recv(_, _, _, dst, dbuf, di)
    # recv-reduce-copy-send - A rrc followed by a send that does not get overwritten
    # rrc(src, sbuf, si, ...) send(_, _, _, dst, dbuf, di)
    def _optimize_rrcs_rrs(self):
        # RRC/S -> RRS
        for slot, ops in self.operations.items():
            frontier = [ops]
            while len(frontier) > 0:
                op = frontier[0]
                if len(op.next) == 1:
                    next_op = op.next[0]
                    if len(next_op.next) == 1:
                        nnext_op = next_op.next[0]
                        if op.inst == Instruction.recv_reduce_copy and next_op.inst == Instruction.send and nnext_op.inst is Instruction.recv and same_tb(op, next_op) and same_count(op, next_op) and same_buf_dst(op, next_op):
                            op.inst = Instruction.recv_reduce_send
                            op.dst = next_op.dst
                            next_op.recv_match.send_match = op
                            op.recv_match = next_op.recv_match
                            remove_op(next_op)

                    if op.inst == Instruction.recv_reduce_copy and next_op.inst == Instruction.send and same_tb(op, next_op) and same_count(op, next_op) and same_buf_dst(op, next_op):
                        op.inst = Instruction.recv_reduce_copy_send
                        op.dst = next_op.dst
                        next_op.recv_match.send_match = op
                        op.recv_match = next_op.recv_match
                        remove_op(next_op)
                frontier = frontier[1:] + op.next

    def _get_tb_step(self, rank, tb):
        if tb in self.tb_steps[rank]:
            self.tb_steps[rank][tb] += 1
            return self.tb_steps[rank][tb]
        else:
            self.tb_steps[rank][tb] = 0
            return 0

    def lower_pt1(self, instances):
        self.infer_dependencies()
        self.lower_buffers(instances)

    def lower_pt2(self, instances, interleaved):
        self.replicate(instances, interleaved)
        return self.lower_tbs()

    def lower_pt2_mscclpp(self, instances, instance_pollicy):
        self.replicate_mscclpp(instances, instance_pollicy)
        return self.lower_tbs()

    def infer_dependencies(self):
        for slot, ops in self.operations.items():
            frontier = [ops]
            while len(frontier) > 0:
                op = frontier[0]
                # Dependencies for every op is the same as the ops that are stored in prev
                # Filter out dependencies that are satisified by tbs executing ops sequentially
                # If multiple dependent ops from the same tb keep the one that happens last
                depends = {}
                for dep_op in list(op.prev):
                    if dep_op.inst != Instruction.start:
                        tb = dep_op.tb
                        if tb not in depends or dep_op.step > depends[tb].step:
                            depends[tb] = dep_op
                op.depends = list(depends.values())
                frontier = frontier[1:] + op.next

    # Convert local scratch buffers to index into one global scratch buffer
    def lower_chunk(self, chunk):
        if chunk is not None and chunk.buffer is not Buffer.input and chunk.buffer is not Buffer.output:
            buffer = self.buffers[chunk.rank][chunk.buffer].get_buffer()
            index = self.buffers[chunk.rank][chunk.buffer].get_global_index(chunk.index)
            return ChunkRef(chunk.rank, buffer, index, chunk.size)
        return chunk

    # Assigns each scratch buffer an offset into the global scratch buffer
    def lower_buffers(self, instances):
        for rank_buffers in self.buffers:
            offset = 0
            for key, buf in rank_buffers.items():
                if key is not Buffer.input and key is not Buffer.output:
                    buf.set_offset(offset)
                    offset += buf.instance_size() * instances

    # Preprocess the threadblocks for lowering into xml
    def lower_tbs(self):
        gpus = []
        for rank, rank_tbs in enumerate(self.instanced_tbs):
            lowered_tbs = {}
            for tbid, tb in rank_tbs.items():
                for op in tb.ops:
                    op.src = self.lower_chunk(op.src)
                    op.dst = self.lower_chunk(op.dst)
                    srcs = sorted(op.srcs, key=lambda x: x[1])
                    dsts = sorted(op.dsts, key=lambda x: x[1])
                    op.srcs = [src[0] for src in srcs]
                    op.dsts = [dst[0] for dst in dsts]
                lowered_tbs[tbid] = tb
            gpus.append(Gpu(rank, list(lowered_tbs.values())))
        return gpus


    # Automatically replicates the algorithm instance number of times
    # interleaved sets the replication policy
    # if True chunks are split as: ChunkA ChunkB -> ChunkA0 ChunkA1 .. ChunkB0 ChunkB1 ...
    # if false chunks are divided as ChunkA0 ChunkB0 ChunkA1 ChunkB1 ...
    # For collectives were chunks are designated for a particular GPU (e.g. AllToAll)
    # only interleaved replication will be correct
    # Interleaved policy only supports single count sends/receives from the input/output buffer
    # (multicount ops are fine between scratch)
    def replicate(self, instances, interleaved):
        if instances == 1:
            self.instanced_tbs = self.tbs
            return

        self.instanced_tbs = []
        for _ in range(self.num_ranks):
            self.instanced_tbs.append({})

        def is_scratch(buffer):
            return buffer != Buffer.input and buffer != Buffer.output

        def get_new_index(rank, buffer, index, size, i):
            # Scratch buffers always use batched
            if is_scratch(buffer):
                buf_instance_len = self.buffers[rank][buffer].instance_size()
                return buf_instance_len * i + index
            # If this is operating on the input/output buffer then replication strategy can be either interleaved or batched
            # This is to fit with the semantics of certain collectives
            elif interleaved:
                return  index * instances + i * size
            else:
                return  len(self.buffers[rank][buffer]) * i + index

        def get_instance_ref(ref):
            iindex = get_new_index(ref.rank, ref.buffer, ref.index, ref.size, i)
            iref = ChunkRef(ref.rank, ref.buffer, iindex, ref.size)
            return iref

        max_channels = max(self.num_channels)
        for i in range(instances):
            # Generate all the threadblocks and ops
            for rank, rank_tbs in enumerate(self.tbs):
                # rank_channels = self.num_channels[rank]
                for tbid, tb in rank_tbs.items():
                    instance_channel = max_channels * i + tb.channel
                    itb = Threadblock(instance_channel, tb.send, tb.recv)
                    itbid = tbid * instances + i
                    itb.ops = [None] * len(tb.ops)
                    for s, op in enumerate(tb.ops):
                        isrc = get_instance_ref(op.src)
                        idst = get_instance_ref(op.dst)
                        idepends = []
                        # Note: We don't need the fill out the rest of the metadata since replication is the last optimization
                        iop = Op(op.inst, op.rank, isrc, idst, idepends, op.step, itbid)
                        itb.ops[s] = iop
                    self.instanced_tbs[op.rank][itbid] = itb

        # Redo dependency analysis
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                for i in range(instances):
                    itbid = tbid * instances + i
                    itb = self.instanced_tbs[rank][itbid]
                    for op, iop in zip(tb.ops, itb.ops):
                        iop.depends = [None] * len(op.depends)
                        for s, dep in enumerate(op.depends):
                            dep_tbid = dep.tb
                            dep_itbid = dep_tbid * instances + i
                            dep_step = dep.step
                            iop.depends[s] = self.instanced_tbs[op.rank][dep_itbid].ops[dep_step]

    def replicate_mscclpp(self, instances, instance_policy):
        # update op step
        for rank, rank_tbs in enumerate(self.tbs):
            for _, tb in rank_tbs.items():
                for id, op in enumerate(tb.ops):
                    op.step = id

        if instances == 1:
            self.instanced_tbs = self.tbs
            return

        self.instanced_tbs = []
        for _ in range(self.num_ranks):
            self.instanced_tbs.append({})

        def is_scratch(buffer):
            return buffer != Buffer.input and buffer != Buffer.output

        def get_new_index(rank, buffer, index, size, i):
            # Scratch buffers always use batched
            if is_scratch(buffer):
                buf_instance_len = self.buffers[rank][buffer].instance_size()
                return buf_instance_len * i + index
            return  len(self.buffers[rank][buffer]) * i + index

        def get_instance_ref(ref):
            iindex = get_new_index(ref.rank, ref.buffer, ref.index, ref.size, i)
            iref = ChunkRef(ref.rank, ref.buffer, iindex, ref.size)
            return iref

        if instance_policy == InstancePolicy.dup:
            for i in range(instances):
                # Generate all the threadblocks and ops
                for rank, rank_tbs in enumerate(self.tbs):
                    # rank_channels = self.num_channels[rank]
                    for tbid, tb in rank_tbs.items():
                        itbid = tbid * instances + i
                        itb = Threadblock(id=itbid)
                        itb.ops = [None] * len(tb.ops)
                        for s, op in enumerate(tb.ops):
                            isrc = get_instance_ref(op.src)
                            idst = get_instance_ref(op.dst)
                            idepends = []
                            # Note: We don't need the fill out the rest of the metadata since replication is the last optimization
                            iop = Op(op.inst, op.rank, isrc, idst, idepends, op.step, itbid, channel_type=op.channel_type)
                            itb.ops[s] = iop
                            for src, step in op.srcs:
                                isrc = get_instance_ref(src)
                                iop.srcs.append((isrc, step))
                            for dst, step in op.dsts:
                                idst = get_instance_ref(dst)
                                iop.dsts.append((idst, step))
                        for chan in tb.channels:
                            itb.channels.append(chan)
                        self.instanced_tbs[op.rank][itbid] = itb

            # Redo dependency analysis
            for rank, rank_tbs in enumerate(self.tbs):
                for tbid, tb in rank_tbs.items():
                    for i in range(instances):
                        itbid = tbid * instances + i
                        itb = self.instanced_tbs[rank][itbid]
                        for op, iop in zip(tb.ops, itb.ops):
                            iop.depends = [None] * len(op.depends)
                            for s, dep in enumerate(op.depends):
                                dep_tbid = dep.tb
                                dep_itbid = dep_tbid * instances + i
                                dep_step = dep.step
                                iop.depends[s] = self.instanced_tbs[op.rank][dep_itbid].ops[dep_step]
