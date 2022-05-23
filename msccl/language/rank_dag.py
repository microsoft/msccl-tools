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
        n.prev =  op.prev.union(n.prev)

def same_tb(op1, op2):
    return op1.tb == op2.tb and op1.channel == op2.channel

def same_count(op1, op2):
    return op1.cnt() == op2.cnt()
    
def same_buf_dst(op1, op2):
    return op1.dst.buffer == op2.dst.buffer and op1.dst.index == op2.dst.index

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
        op = Op(Instruction.copy, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel=ch)
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
        op = Op(Instruction.reduce, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel=ch)
        dstbuffer = recv_ref.buffer
        dstindex = recv_ref.index
        srcbuffer = send_ref.buffer
        srcindex = send_ref.index
        size = recv_ref.size
        prev_ops = []
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

    def convert_set_list(self):
        ops = []
        for slot, op in self.operations.items():
            if op.inst == Instruction.start:
                op.next = list(op.next)
                for o in op.next:
                    ops.append(o)
            elif op.inst != Instruction.copy:
                ops.append(op)

            visited = set()
            while len(ops) > 0:
                op = ops[0]
                if op not in visited:
                    visited.add(op)
                    op.next = list(op.next)
                    ops = ops[1:] + op.next
                else:
                    ops = ops[1:]
                    
    def optimize(self):
        self._optimize_rrcs_rrs()
        self._optimize_rcs()

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

    def lower_pt1(self, instances):
        self.infer_dependencies()
        self.lower_buffers(instances)
    
    def lower_pt2(self, instances, interleaved):
        self.replicate(instances, interleaved)
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
        if chunk.buffer is not Buffer.input and chunk.buffer is not Buffer.output:
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

