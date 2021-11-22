# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from enum import Enum
import heapq

from sccl.language.ir import *
from sccl.language.passes import *

def not_send(op, slot):
    # If the instruction on this spot is a copy or reduce, check to see if the slot was the dst of the operation
    if op.inst == Instruction.copy or op.inst == Instruction.reduce:
        cpy_src = op.src
        buffer, index = slot
        return buffer != cpy_src.buffer or (index < cpy_src.index and index > (cpy_src.index + cpy_src.size))
    return op.inst != Instruction.send

def remove_op(op):
    for p in op.prev:
        p.next.remove(op)
        p.next = op.next.union(p.next)

    for n in op.next:
        n.prev.remove(op)
        n.prev =  op.prev.union(n.prev)

def same_tb(op1, op2):
    return op1.tb == op2.tb

def same_count(op1, op2):
    return op1.cnt() == op2.cnt()
    

class RankDAG:
    def __init__(self, rank, buffers):
        self.rank = rank
        self.buffers = buffers
        self.slots = []
        self.operations = {} # slot -> operations
        self.send_ranks = set() # Set of ranks this rank sends to
        self.recv_ranks = set() # Set of ranks this rank receives from
        self.tbs = {}
        self.tb_mapping = {}
        self.send_channel_mapping = {} # sender rank -> channels -> tbid
        self.recv_channel_mapping = {} 

    def add_start(self, buffer, index, ref):
        slot = (buffer, index)
        self.slots.append(slot)

        op = Op(Instruction.start, self.rank, ref, ref, next=set(), prev=set())
        self.operations[slot] = op

    # Find the last write to happen on this slot
    def find_last_recv(self, slot):
        def dfs(op):
            # Found the last operation on the slot
            if len(op.next) == 0:
                return not_send(op, slot), op
            else:
                last_recvs = False
                # Check if any of the children is the last write
                for o in op.next:
                    is_last_recv, recv_op = dfs(o)
                    if is_last_recv:
                        return True, recv_op
                    last_recvs = last_recvs or is_last_recv
                # Check if we are the last write
                if not_send(op, slot) and not last_recvs:
                    return True, op
                return False, op
                
        result, op = dfs(self.operations[slot])
        assert result
        return op

    # Find the last set of operations that happened on this slot
    # There may be multiple as sends can happen in parallel
    def find_last_ops(self, slot_ops):
        frontier = [slot_ops]
        last_ops = []

        while len(frontier) > 0:
            op = frontier[0]
            if len(op.next) == 0:
                last_ops.append(op)
            frontier = frontier[1:] + list(op.next)   
        return last_ops

    def add_copy(self, send_ref, recv_ref, step, priority, tb):
        op = Op(Instruction.copy, self.rank, send_ref, recv_ref, chunk_step=step, priority=priority, next=set(), prev=set(), tb=tb)
        dstbuffer = recv_ref.buffer
        dstindex = recv_ref.index
        srcbuffer = send_ref.buffer
        srcindex = send_ref.index
        size = recv_ref.size
        prev_ops = []

        # Sending part of copy
        for i in range(srcindex, srcindex+size):
            slot = (srcbuffer, i)
            prev_op = self.find_last_recv(slot) # All operations that need to happen before
            prev_op.next.add(op)
            op.prev.add(prev_op)

        # Receiving part of copy
        for i in range(dstindex, dstindex+size):
            slot = (dstbuffer, i)
            if slot in self.operations:
                slot_ops = self.operations[slot]
                prev_ops = self.find_last_ops(slot_ops) # All operations that need to happen before

                if len(prev_ops) > 0:
                    for prev_op in prev_ops:
                        prev_op.next.add(op)
                        op.prev.add(prev_op)
            else:
                self.operations[slot] = op

    def add_reduce(self, send_ref, recv_ref, step, priority, tb):
        op = Op(Instruction.reduce, self.rank, send_ref, recv_ref, chunk_step=step, priority=priority, next=set(), prev=set(), tb=tb)
        dstbuffer = recv_ref.buffer
        dstindex = recv_ref.index
        srcbuffer = send_ref.buffer
        srcindex = send_ref.index
        size = recv_ref.size
        prev_ops = []

        # B
        for i in range(srcindex, srcindex+size):
            slot = (srcbuffer, i)
            prev_op = self.find_last_recv(slot) # All operations that need to happen before
            prev_op.next.add(op)
            op.prev.add(prev_op)

        # A
        for i in range(dstindex, dstindex+size):
            slot = (dstbuffer, i)
            if slot in self.operations:
                slot_ops = self.operations[slot]
                prev_ops = self.find_last_ops(slot_ops) # All operations that need to happen before

                if len(prev_ops) > 0:
                    for prev_op in prev_ops:
                        prev_op.next.add(op)
                        op.prev.add(prev_op)
            else:
                self.operations[slot] = op

    def add_send(self, send_ref, recv_ref, step, priority, tb, ch):
        self.send_ranks.add(recv_ref.rank)
        op = Op(Instruction.send, self.rank, send_ref, recv_ref, chunk_step=step, priority=priority, next=set(), prev=set(), tb=tb, channel=ch)
        buffer = send_ref.buffer
        index = send_ref.index
        size = send_ref.size
        prev_ops = []
        for i in range(index, index+size):
            slot = (buffer, i)
            prev_op = self.find_last_recv(slot)
            prev_ops.append(prev_op) # All operations that need to happen before

        for prev_op in prev_ops:
            if op not in prev_op.next:
                prev_op.next.add(op)
                op.prev.add(prev_op)
        return op

    def add_recv(self, send_ref, recv_ref, step, priority, tb, ch):
        self.recv_ranks.add(send_ref.rank)
        op = Op(Instruction.recv, self.rank, send_ref, recv_ref, chunk_step=step, priority=priority, next=set(), prev=set(), tb=tb, channel=ch)
        buffer = recv_ref.buffer
        index = recv_ref.index
        size = recv_ref.size

        for i in range(index, index+size):
            slot = (buffer, i)

            if slot in self.operations:
                slot_ops = self.operations[slot]
                prev_ops = self.find_last_ops(slot_ops) # All operations that need to happen before
                if len(prev_ops) > 0:
                    for prev_op in prev_ops:
                        prev_op.next.add(op)
                        op.prev.add(prev_op)
            else:
                self.operations[slot] = op
        return op

    def add_recv_reduce_copy(self, send_ref, recv_ref, step, priority, tb, ch):
        self.recv_ranks.add(send_ref.rank)
        op = Op(Instruction.recv_reduce_copy, self.rank, send_ref, recv_ref, chunk_step=step, priority=priority, next=set(), prev=set(), tb=tb, channel=ch)
        buffer = recv_ref.buffer
        index = recv_ref.index
        size = recv_ref.size

        for i in range(index, index+size):
            slot = (buffer, i)
            if slot in self.operations:
                slot_ops = self.operations[slot]
                prev_ops = self.find_last_ops(slot_ops) # All operations that need to happen before

                if len(prev_ops) > 0:
                    for prev_op in prev_ops:
                        prev_op.next.add(op)
                        op.prev.add(prev_op)
            else:
                self.operations[slot] = op
        return op

    def optimize(self):
        self._optimize_rrcs_rrs()
        self._optimize_rcs()
        
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
                if len(op.next) == 1:
                    next_op = list(op.next)[0] # TODO: FIX ME
                    if op.inst == Instruction.recv and next_op.inst == Instruction.send and same_tb(op, next_op) and same_count(op, next_op):
                        op.inst = Instruction.recv_copy_send
                        op.dst = next_op.dst
                        op.match = op.match + next_op.match
                        remove_op(next_op)
                frontier = frontier[1:] + list(op.next)
        
    def _optimize_rrcs_rrs(self):
        # RRC/S -> RRS
        for slot, ops in self.operations.items():
            frontier = [ops]
            while len(frontier) > 0:
                op = frontier[0]
                if len(op.next) == 1:
                    next_op = list(op.next)[0] # TODO: FIX ME
                    if len(next_op.next) == 1:
                        nnext_op = list(next_op.next)[0] # TODO: FIX ME
                        if op.inst == Instruction.recv_reduce_copy and next_op.inst == Instruction.send and nnext_op.inst == Instruction.recv and same_tb(op, next_op) and same_count(op, next_op):
                            op.inst = Instruction.recv_reduce_send
                            op.dst = next_op.dst
                            op.match = op.match + next_op.match
                            remove_op(next_op)
                    
                    if op.inst == Instruction.recv_reduce_copy and next_op.inst == Instruction.send and same_tb(op, next_op) and same_count(op, next_op):
                        op.inst = Instruction.recv_reduce_copy_send
                        op.dst = next_op.dst
                        op.match = op.match + next_op.match
                        remove_op(next_op)
                frontier = frontier[1:] + list(op.next)


    def verify_tb_op_compatible(self, tb, op):
        s = op.dst.rank if op.is_send() else -1
        r = op.src.rank if op.is_recv() else -1
            
        sends_ok = tb.send == s or s == -1 or tb.send == -1
        recvs_ok = tb.recv == r or r == -1 or tb.recv == -1
        channel_ok = tb.channel == op.channel or tb.channel == -1 or op.channel == -1
        return sends_ok and recvs_ok and channel_ok

    def add_tb_op(self, tbid, op):
        if tbid not in self.tbs:
            self.tbs[tbid] = Threadblock()
        tb = self.tbs[tbid]
        if self.verify_tb_op_compatible(tb, op):
            tb.ops.append(op)
            tb.channel = op.channel
            tb.send = op.dst.rank if op.is_send() else tb.send
            tb.recv = op.src.rank if op.is_recv() else tb.recv
            op.step = len(tb.ops)-1
        else:
            print("Illegal TB assignment")
            print("TODO: Add Debug messages")
            sys.exit()

    # Manual threadblock, channel assignment
    def assign_tbs(self):
        ops = []
        for slot, op in self.operations.items():
            if op.inst == Instruction.start:
                for o in list(op.next):
                    heapq.heappush(ops, o)
            elif op.inst != Instruction.copy:
                heapq.heappush(ops, op)

        visited = set()
        while len(ops) > 0:
            op = heapq.heappop(ops)
            if op not in visited:
                visited.add(op)
                self.add_tb_op(op.tb, op)
                
                for o in list(op.next):
                    heapq.heappush(ops, o)

    def get_tb_options(self, dict, key, assigned_tbs, unassigned):
        if key == -1: # Can go anywhere
            x = set(list(i for i in range(0, assigned_tbs)))
            return x
        elif key in dict:
            x = dict[key].union(unassigned)
            return x
        else:
            return unassigned
    
    def _get_channel(self, send, recv):
        channel = 0
        for tb in self.tbs.values():
            if (tb.send == send and tb.send != -1 and send != -1) \
             or (tb.recv == recv and tb.recv != -1 and recv != -1):
                channel = max(tb.channel + 1, channel)
        return channel

    def add_set(self, dict, key, value):
        if key in dict:
            dict[key].add(value)
        else:
            dict[key] = set([value])

    # Heuristic: Assign based 1) Operations that are ready
    # 2) Priority of operation: cost function is steps until end
    def auto_assign_tbs(self, num_tb):
        recvs2tbch = {}
        sends2tbch = {}
        next_unassigned_tb = 0
        current_tb_step = {}
        unassigned_recv = set()
        unassigned_send = set()

        ops = []
        for slot, op in self.operations.items():
            if op.inst == Instruction.start:
                for o in list(op.next):
                    heapq.heappush(ops, o)
            elif op.inst != Instruction.copy:
                heapq.heappush(ops, op)

        visited = set()
        while len(ops) > 0:
            op = heapq.heappop(ops)
            if op not in visited:
                visited.add(op)
                
                s = op.dst.rank if op.is_send() else -1
                r = op.src.rank if op.is_recv() else -1
                # Get all possible TBs this can be mapped to
                recv_opts = self.get_tb_options(recvs2tbch, r, next_unassigned_tb, unassigned_recv)
                send_opts = self.get_tb_options(sends2tbch, s, next_unassigned_tb, unassigned_send)
                tb_options = recv_opts.intersection(send_opts)
                # Can't map to existing TB - use a new one
                if len(tb_options) == 0:
                    if next_unassigned_tb >= num_tb and num_tb!=0:
                        print("Too few threadblocks:",  num_tb)
                        for tb in self.tbs.values():
                            print("TB send", tb.send, "recv", tb.recv, "channel", tb.channel)
                        sys.exit()
                    tbid = next_unassigned_tb
                    next_unassigned_tb += 1
                    channel = 0 # First TB with this combination of s/r
                    self.tbs[tbid] = Threadblock(send=s, recv=r, channel=channel)
                    current_tb_step[tbid] = 0
                else: 
                    tbid = list(tb_options)[0]
                    for tbid_opt in tb_options:
                        if current_tb_step[tbid_opt] < current_tb_step[tbid]:
                            tbid = tbid_opt
                    if op.chunk_step <= current_tb_step[tbid] and op.priority > 0 \
                        and (next_unassigned_tb < num_tb or num_tb==0):
                        tbid = next_unassigned_tb
                        next_unassigned_tb += 1
                        channel = self._get_channel(s, r)
                        self.tbs[tbid] = Threadblock(send=s, recv=r, channel=channel)
                        current_tb_step[tbid] = 0
                # if not self.verify_tb_op_compatible(self.tbs[tbid], op):
                #     print("Failing :(", tbid, op, op.channel)
                    
                self.add_set(recvs2tbch, r, tbid)
                self.add_set(sends2tbch, s, tbid)

                tb = self.tbs[tbid]
                tb.ops.append(op)
                tb.send = op.dst.rank if op.is_send() else tb.send
                tb.recv = op.src.rank if op.is_recv() else tb.recv
                
                op.step = len(tb.ops)-1
                op.channel = tb.channel
                op.tb = tbid
                current_tb_step[tbid] = op.chunk_step
                if tb.recv == -1:
                    unassigned_recv.add(tbid)
                elif tbid in unassigned_recv:
                    unassigned_recv.remove(tbid)
                if tb.send == -1:
                    unassigned_send.add(tbid)
                elif tbid in unassigned_send:
                    unassigned_send.remove(tbid)
                # For correctness make certain the matching sends and receives
                # happen on the same channel
                for match in op.match:
                    match.channel = tb.channel

                for o in list(op.next):
                    heapq.heappush(ops, o)

        # for tbid, tb in self.tbs.items():
        #     print("TB:", tbid, "s", tb.send, "r", tb.recv)
        #     for op in tb.ops:
        #         print("  ", op.chunk_step, op.priority, op)

    def lower_pt1(self, instances):
        self.infer_dependencies()
        check_dependency_cycles(self.tbs)
        self.lower_buffers(instances)
    
    def lower_pt2(self, instances, buffers, interleaved):
        self.replicate(instances, buffers, interleaved)
        return self.lower_tbs(buffers)


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
                frontier = frontier[1:] + list(op.next)

    # Convert local scratch buffers to index into one global scratch buffer
    def lower_chunk(self, chunk, buffers):
        if chunk.buffer is not Buffer.input and chunk.buffer is not Buffer.output:
            buffer = buffers[chunk.rank][chunk.buffer].get_buffer()
            index = buffers[chunk.rank][chunk.buffer].get_global_index(chunk.index)
            return ChunkRef(chunk.rank, buffer, index, chunk.size)
        return chunk

    # Assigns each scratch buffer an offset into the global scratch buffer
    def lower_buffers(self, instances):
        offset = 0
        for key, buf in self.buffers.items():
            if key is not Buffer.input and key is not Buffer.output:
                buf.set_offset(offset)
                offset += buf.instance_size() * instances

    # Preprocess the threadblocks for lowering into xml
    def lower_tbs(self, buffers):
        for tb in self.instanced_tbs.values():
            for op in tb.ops:
                op.src = self.lower_chunk(op.src, buffers)
                op.dst = self.lower_chunk(op.dst, buffers)
        return Gpu(self.rank, self.instanced_tbs.values())


    # Automatically replicates the algorithm instance number of times
    # interleaved sets the replication policy
    # if True chunks are split as: ChunkA ChunkB -> ChunkA0 ChunkA1 .. ChunkB0 ChunkB1 ...
    # if false chunks are divided as ChunkA0 ChunkB0 ChunkA1 ChunkB1 ...
    # For collectives were chunks are designated for a particular GPU (e.g. AllToAll) 
    # only interleaved replication will be correct
    # Interleaved policy only supports single count sends/receives from the input/output buffer
    # (multicount ops are fine between scratch)
    def replicate(self, instances, buffers, interleaved):
        if instances == 1:
            self.instanced_tbs = self.tbs
            return 

        self.instanced_tbs = {}

        def is_scratch(buffer):
            return buffer != Buffer.input and buffer != Buffer.output

        def get_new_index(rank, buffer, index, size, i):
            # Scratch buffers always use batched
            if is_scratch(buffer):
                buf_instance_len = buffers[rank][buffer].instance_size()
                return buf_instance_len * i + index
            # If this is operating on the input/output buffer then replication strategy can be either interleaved or batched
            # This is to fit with the semantics of certain collectives
            elif interleaved:
                return  index * instances + i * size
            else:
                return  len(buffers[rank][buffer]) * i + index

        def get_instance_ref(ref):
            iindex = get_new_index(ref.rank, ref.buffer, ref.index, ref.size, i)
            iref = ChunkRef(ref.rank, ref.buffer, iindex, ref.size)
            return iref

        for i in range(instances):
            # Generate all the threadblocks and ops
            for tbid, tb in self.tbs.items():
                # TODO: Handle channels correctly
                itb = Threadblock(tb.channel+i, tb.send, tb.recv)
                itbid = tbid * instances + i
                itb.ops = [None] * len(tb.ops)
                for s, op in enumerate(tb.ops):
                    isrc = get_instance_ref(op.src)
                    idst = get_instance_ref(op.dst)
                    idepends = [] 
                    iop = Op(op.inst, op.rank, isrc, idst, idepends, op.step, itbid) # Note: We don't need the fill out the rest of the metadata
                    itb.ops[s] = iop
                self.instanced_tbs[itbid] = itb
        
        # Redo dependency analysis
        for tbid, tb in self.tbs.items():
            for i in range(instances):
                itbid = tbid * instances + i
                itb = self.instanced_tbs[itbid]
                for op, iop in zip(tb.ops, itb.ops):
                    iop.depends = [None] * len(op.depends)
                    for s, dep in enumerate(op.depends):
                        dep_tbid = dep.tb
                        dep_itbid = dep_tbid * instances + i
                        dep_step = dep.step
                        iop.depends[s] = self.instanced_tbs[dep_itbid].ops[dep_step] 

