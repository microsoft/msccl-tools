# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from dataclasses import dataclass
from enum import Enum
from sccl.language.ir import *
from sccl.language.misc import *
from sccl.language.passes import *
import sccl.collectives as collectives

_current_program = None
def _curr():
    global _current_program
    if _current_program == None:
        raise RuntimeError("No Program in context")
    return _current_program

class SCCLProgram:
    def __init__(self, name, topo, collective, instances, protocol='Simple'):
        self.name = name
        self.topo = topo
        self.collective = collective       
        self.ranks = []
        self.instances = instances
        self.protocol = protocol
        assert protocol == 'Simple' or protocol == 'LL' or protocol == 'LL128', \
            f'Given protocol: {protocol}. Must be either Simple, LL, LL128'
        self.run_opt = True # Runs optimization passes
        # Initialize the input buffers
        num_ranks = topo.num_nodes()
        rank_buffers = collective.init_buffers()
        for r in range(num_ranks):
            self.ranks.append(Process(self, r, rank_buffers[r]))

    # Returns a Process corresponding to rank number
    def rank(self, rank):
        return self.ranks[rank]

    # Checks that all chunks that should be on each rank
    # are present in the output buffer.
    def check(self):
        return self.collective.check(self)

    # Lower program to XML
    def lower(self):
        for rank in self.ranks:
            if self.run_opt:
                rank.optimize()
            rank.reorder_ops()
            rank.detect_dependencies()
            rank.lower_buffers()
        for rank in self.ranks:
            check_dependency_cycles(rank.tbs)
            check_threadblock_ordering(rank.tbs, self.ranks)
        gpu_prgms = [rank.lower_tbs() for rank in self.ranks]
        return Program(self.name, self.collective.name, self.collective.inplace, self.protocol, gpu_prgms)

    def __enter__(self):
        global _current_program
        if _current_program != None:
            raise RuntimeError("There is already a SCCL Program in context")
        _current_program = self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _current_program
        if _current_program != self:
            raise RuntimeError("This program is not currently in context")
        _current_program = None

def Rank(index):
    return _curr().rank(index)

def XML():
   print(ir_to_xml(_curr().lower()))

def Check():
    return _curr().check()

# Scratch buffer slice with manual indexing
class BufferSlice:
    def __init__(self, buf):
        self.buf = buf
        self.offset = -1 # Offset into the global scratch buffer
        self.index = 0 # Current index of chunks added
        self.chunks = {}

    # Returns the global index into the scratch buffer
    def get_global_index(self, index):
        assert (self.offset > -1), 'set_offset needs to be called first'
        return self.offset + index

    def get_buffer(self):
        return self.buf

    def size(self):
        return len(self.chunks)

    def set_offset(self, offset):
        self.offset = offset

    def get_next_index(self, size):
        index = self.index
        self.index += size
        return index

    def __getitem__(self, index):
        return self.chunks[index]
    
    def __setitem__(self, index, value):
        self.chunks[index] = value


class Process:
    def __init__(self, prog, rank, buffers):
        self.prog = prog
        self.rank = rank
        self.buffers = buffers
        self.tbs = {}
        self.tb_mapping = {}
        self.send_channel_mapping = {} # sender rank -> channels -> tbid
        self.recv_channel_mapping = {} # receiver rank -> channels -> tbid
        self.tb_count = 0
        self.slot_ops = {} # (buffer, idx) -> list of ops on this slot

    def _get_tbid(self, inst, other_rank, ch):
        key = (inst, other_rank, ch)
        if key in self.tb_mapping:
            tbid = self.tb_mapping[key]
        else: # TODO: Currently can mix default tb assignment and manual tb assignment which will break
            self.tb_mapping[key] = self.tb_count
            tbid = self.tb_count
            self.tb_count += 1
        return tbid

    def _tb_addop(self, tbid, sendto, receivefrom, ch, op):
        def check_tb_channels():
            # Check this channel isn't being used by another tb to send to the same rank
            if sendto != -1:
                if sendto not in self.send_channel_mapping:
                    self.send_channel_mapping[sendto] = {ch:tbid}
                else:
                    senders_channels = self.send_channel_mapping[sendto]
                    assert ch not in senders_channels or senders_channels[ch] == tbid, \
                        f'Cannot assign threadblock {tbid} to channel {ch}. Already being used by threadblock {senders_channels[ch]} to send to rank {sendto}'
                    self.send_channel_mapping[sendto][ch] = tbid
                
            # Check this channel isn't being used by another tb to receive from the same rank
            if receivefrom != -1:
                if receivefrom not in self.recv_channel_mapping:
                    self.recv_channel_mapping[receivefrom] = {ch:tbid}
                else:
                    receivers_channels = self.recv_channel_mapping[receivefrom]
                    assert ch not in receivers_channels or receivers_channels[ch] == tbid, \
                        f'Cannot assign threadblock {tbid} on channel {ch}. Already being used by threadblock {receivers_channels[ch]} to receive from {receivefrom}'
                    self.recv_channel_mapping[receivefrom][ch] = tbid
            
        # Create a new threadblock if tbid doesn't exist
        if tbid not in self.tbs:
            check_tb_channels()
            self.tbs[tbid] = Threadblock(ch, sendto, receivefrom, ops=[op])

        # Check it is valid to use this threadblock if already exists
        else:
            tb = self.tbs[tbid]
            assert (sendto == -1 or tb.send == -1 or tb.send == sendto), \
                f'Rank {self.rank}: Threadblock {tbid} is already set to send to {tb.send}, trying to send to {sendto}'
            assert (receivefrom == -1 or tb.recv == -1 or tb.recv == receivefrom), \
                   f'Rank {self.rank}: Threadblock {tbid} is already set to receive from {tb.recv}, trying to receive from {receivefrom}'
            check_tb_channels()
            # Update existing threadblocks if adding a sender/receiver when one previously didn't exist
            if tb.send == -1 and sendto != -1:
                tb.send = sendto
            if tb.recv == -1 and receivefrom != -1:
                tb.recv = receivefrom
            tb.ops.append(op)


    def _add_send(self, tbid, ch, op):
        assert(op.inst == Instruction.send)
        sendto = op.dst.rank

        # Update tb and assign to a default tb if not given
        if tbid == -1:
            tbid = self._get_tbid(Instruction.send, sendto, ch)

        self._tb_addop(tbid, sendto, -1, ch, op)

        # Fill in op fields 
        op.tb = tbid
        op.step = len(self.tbs[tbid].ops)-1
        
        # Update slot_ops 
        for i in range(op.src.size):
            self._update_slot_ops(op.src.buffer, op.src.index+i, op, tbid)

        
    def _add_recv(self, tbid, ch, op):
        assert(op.inst == Instruction.recv)
        receivefrom = op.src.rank
        if tbid == -1:
            tbid = self._get_tbid(Instruction.recv, receivefrom, ch)
        recvd_chunkref = op.dst

        # Update tbs
        self._tb_addop(tbid, -1, receivefrom, ch, op)

        # Fill in op fields
        op.tb = tbid
        op.step = len(self.tbs[tbid].ops)-1

         # Update buffer with received chunks, update dependencies for those chunks
        for i in range(op.src.size):
            self.buffers[op.dst.buffer][op.dst.index+i] = self.prog.ranks[op.src.rank].buffers[op.src.buffer][op.src.index+i]
            self._update_slot_ops(op.dst.buffer, op.dst.index+i, op, tbid)

    def _add_copy(self, tbid, ch, op):
        assert(op.inst == Instruction.copy)
        if tbid == -1:
            tbid = self._get_tbid(Instruction.copy, -1, ch)
        
        # Update tbs
        self._tb_addop(tbid, -1, -1, ch, op)
        
        # Fill in op fields 
        op.tb = tbid
        op.step = len(self.tbs[tbid].ops)-1

        # Update buffer copied chunks and dependencies
        for i in range(op.src.size):
            self.buffers[op.dst.buffer][op.dst.index+i] = self.buffers[op.src.buffer][op.src.index+i]
            self._update_slot_ops(op.dst.buffer, op.dst.index+i, op, tbid)
            self._update_slot_ops(op.src.buffer, op.src.index+i, op, tbid)

    def _add_receive_reduce_copy(self, tbid, ch, op):
        receivefrom = op.src.rank
        if tbid == -1:
            tbid = self._get_tbid(Instruction.recv, receivefrom, ch) 
        recvd_chunkref = op.dst

        # Update tbs
        self._tb_addop(tbid, -1, receivefrom, ch, op)

        # Fill in op fields 
        op.tb = tbid
        op.step = len(self.tbs[tbid].ops)-1

        # Update buffer with reduced chunks and dependencies for each chunk
        for i in range(op.src.size):
            reduce_chunk = self.buffers[op.dst.buffer][op.dst.index+i]
            other_chunk = self.prog.ranks[op.src.rank].buffers[op.src.buffer][op.src.index+i]
            self.buffers[op.dst.buffer][op.dst.index+i] = reduce_chunk.reduce(other_chunk)
            self._update_slot_ops(op.dst.buffer, op.dst.index+i, op, tbid)

    def _add_reduce(self, tbid, ch, op):
        receivefrom = op.src.rank
        if tbid == -1:
            tbid = self._get_tbid(Instruction.copy, receivefrom, ch)
        recvd_chunkref = op.dst

        # Update buffer with reduced chunk and dependences
        for i in range(op.src.size):
            reduce_chunk = self.buffers[op.dst.buffer][op.dst.index+i]
            other_chunk = self.prog.ranks[op.src.rank].buffers[op.src.buffer][op.src.index+i]
            self.buffers[op.dst.buffer][op.dst.index+i] = other_chunk.reduce(reduce_chunk)
            self._update_slot_ops(op.dst.buffer, op.dst.index+i, op, tbid)
            self._update_slot_ops(op.src.buffer, op.src.index+i, op, tbid)

        # Update tbs
        self._tb_addop(tbid, -1, -1, ch, op)

        # Fill in op fields 
        op.tb = tbid
        op.step = len(self.tbs[tbid].ops)-1

    def _update_slot_ops(self, buffer, index, op, tbid):
        key = (buffer, index)
        if key in self.slot_ops:
            self.slot_ops[key].append(op)
        else:
            self.slot_ops[key] = [op]

    def get_ref(self, buffer, index, size):
        creator = []
        for i in range(index, index+size):
            if (buffer, i) in self.slot_ops:
                creator_index = len(self.slot_ops[(buffer, i)]) - 1
                op = self.slot_ops[(buffer, i)][creator_index]
                while op.inst == Instruction.send and creator_index > 0:
                    creator_index -= 1
                    op = self.slot_ops[(buffer, i)][creator_index]
                if op.inst != Instruction.send:
                    creator.append(op)
        return Ref(buffer, index, size, self.prog, self.rank, creator=creator)

    # Returns a reference to the chunk located at index of the input buffer.
    def input(self, index, size=1):
        buffer, index = self.prog.collective.get_buffer_index(self.rank, Buffer.input, index)
        return self.get_ref(buffer, index, size)

    def output(self, index, size=1):
        buffer, index = self.prog.collective.get_buffer_index(self.rank, Buffer.output, index)
        return self.get_ref(buffer, index, size)

    def scratch(self, name, index, size=1):
        return self.get_ref(name, index, size)

    # Creates a scratch buffer with a name
    def create_scratch(self, name):
        assert (name not in self.buffers), f'Scratch buffer, {name}, already created'
        self.buffers[name] = BufferSlice(Buffer.scratch)

    # Runs optimization pass over slot_ops
    # TODO: This is now doing more than optimization...
    def optimize(self):
        for k, ops in self.slot_ops.items():
            rrcs_rrs(ops, self.tbs)
            rcs(ops, self.tbs)
        # Delete ops that are no longer needed
        # Instruction reordering within tbs
        for tbid, tb in self.tbs.items():
            delete_pass(tb)

    def reorder_ops(self):
        rerun = False
        for tbid, tb in self.tbs.items():
            rerun |= self.order_ops_in_tb(tbid)
        if rerun:
            self.reorder_ops()
    
    def detect_dependencies(self):
        for _, ops in self.slot_ops.items():
            update_slot_dependency(ops)
        for _, ops in self.slot_ops.items():
            for op in ops:
                if type(op.depends) is dict:
                    op.depends = list(op.depends.values())

    # If chunks are sent from one threadblock x to threadblock y they are received in the same order
    # This is required to avoid deadlock. 
    # tb:x send(a), send(b) -> tb:y recv(a), recv(b)
    # Currently the order is defined by programmer, but we need place RRCS/RRS/RCS instructions
    # appropriately to avoid deadlock
    # Returns whether or not operations were re-ordered
    # TODO: Eventually want to order operations automatically based off priority
    # TODO: Does this terminate?
    def order_ops_in_tb(self, tbid):
        tb = self.tbs[tbid]
        other_tbs = {} # TBs this TB communicates with (r, tb) -> step of last op
        # Gather all unique TBs this TB communicates with
        rerun = False
        for op in tb.ops:
            for next in op.next:
                other_tbs[(next.rank, next.tb)] = (-1, -1)
        # Check that the ordering of everything is preserved
        for op in tb.ops:
            this_current_step = op.step
            for next in op.next:
                other_current_step = next.step
                other_last_step, this_last_step = other_tbs[(next.rank, next.tb)]
                if other_last_step > other_current_step:
                    # Reorder this current op (step, op) to (prev_tb_step, op)
                    tb.ops[this_current_step] = tb.ops[this_last_step]
                    tb.ops[this_last_step] = op
                    tb.ops[this_current_step].step = this_current_step
                    tb.ops[this_last_step].step = this_last_step
                    return True
                other_tbs[(next.rank, next.tb)] = (other_current_step, this_current_step)
        return False

    # Convert local scratch buffers to index into one global scratch buffer
    def lower_chunk(self, chunk):
        if chunk.buffer is not Buffer.input and chunk.buffer is not Buffer.output:
            rank = self.prog.ranks[chunk.rank]
            buffer = rank.buffers[chunk.buffer].get_buffer()
            index = rank.buffers[chunk.buffer].get_global_index(chunk.index)
            return ChunkRef(buffer, index, chunk.size)
        return chunk

    # Assigns each scratch buffer an offset into the global scratch buffer
    def lower_buffers(self):
        offset = 0
        for key, buf in self.buffers.items():
            if key is not Buffer.input and key is not Buffer.output:
                buf.set_offset(offset)
                offset += buf.size()

    # Preprocess the threadblocks for lowering into xml
    def lower_tbs(self):
        for tb in self.tbs.values():
            # Sort Ops by step
            # Index scratch buffers
            for op in tb.ops:
                op.src = self.lower_chunk(op.src)
                op.dst = self.lower_chunk(op.dst)
        return Gpu(self.rank, self.tbs.values())

@dataclass
class Chunk:
    origin_rank: int # Rank the chunk initially started at
    origin_index: int # Index the chunk initially started at
    dst_rank: int = -1
    dst_index: int = -1

    def reduce(self, chunk):
        if type(chunk) is ReduceChunk:
            return chunk.reduce(self)
        elif type(chunk) is Chunk:  
            chunks = [self, chunk]
            return ReduceChunk(chunks)
        else:
            assert True, "Trying to reduce with chunk of None"
            return None

    def __eq__(self, other):
        return type(other) is Chunk and self.origin_rank == other.origin_rank and self.origin_index == other.origin_index

    def __lt__(self, other):
        return self.origin_rank < other.origin_rank or \
               (self.origin_rank == other.origin_rank and self.origin_index < other.origin_index)

@dataclass
class ReduceChunk:
    chunks: list # List of chunks reduced

    def reduce(self, chunk):
        if type(chunk) is ReduceChunk:
            chunks = self.chunks + chunk.chunks
        elif type(chunk) is Chunk:  
            chunks =self.chunks + [chunk]
        else:
            assert True, "Trying to reduce with chunk of None"
        return ReduceChunk(chunks)

    def sort(self):
        self.chunks.sort()

    # Two reduce chunks are equal if they contain the same list of
    # chunks being reduced
    def __eq__(self, other):
        self.sort()
        other.sort()
        return self.chunks == other.chunks


@dataclass
class Ref(ChunkRef):
    prog: SCCLProgram
    rank: int
    missing: set = field(default_factory=set)
    creator: list = field(default_factory=list) # The operation(s) that created this chunk on this rank

    def __repr__(self):
        return f'Ref(Buffer:{self.buffer}, Index:{self.index}, Size:{self.size}, Rank:{self.rank})'

    def _end(self):
        return self.index + self.size

    def _get_chunk(self, index):
        return self.prog.ranks[self.rank].buffers[self.buffer][index]

    def _copy(self, buffer=Buffer.output, index=-1, tb=-1, ch=0):
        dst_chunkref = self.prog.ranks[self.rank].get_ref(buffer, index, self.size)
        op = Op(Instruction.copy, self.rank, self, dst_chunkref, {})

        op.prev = self.creator
        for p in op.prev:
            p.next.append(op)

        self.prog.ranks[self.rank]._add_copy(tb, ch, op)
        return dst_chunkref

    def split(self, num):
        assert (self.size % num == 0), f'Trying to split a chunk of {self.size} elements into {num} parts'
        chunks = [None] * num
        size = self.size // num
        for i in range(num):
            index = self.index + i * size
            chunks[i] = self.prog.ranks[self.rank].get_ref(self.buffer, index, size)
        return chunks

    # TODO: this is weird...
    def group(self, other):
        assert (self.rank == other.rank), f'Trying to concatenate chunks on ranks {self.rank} and {other.rank}'
        assert (self.buffer == other.buffer), f'Trying to concatenate chunks in {self.buffer} and {other.buffer}'
        if self.index < other.index:
            first = self
            second = other
        else:
            first = other
            second = self

        end = max(first._end(), second._end())
        missing = set(range(first.index, end))
        missing.difference_update(set(range(first.index, first._end())).difference(first.missing))
        missing.difference_update(set(range(second.index, second._end())).difference(second.missing))
        return Ref(self.buffer, first.index, end - first.index, self.prog, self.rank, missing) # Broken
        

    def send(self, dst, buffer=None, index=-1, sendtb=-1, recvtb=-1, ch=0):
        assert (len(self.missing) == 0), f'Trying to send an incomplete concatenation. Missing indices {self.missing}'

        # If index is not specified assume it is going to the same place in the next gpu
        if index == -1 and buffer == None:
            index = self.index
            buffer = self.buffer
        elif index == -1 and buffer is not Buffer.input and buffer is not Buffer.output:
            index = self.prog.ranks[dst].buffers[buffer].get_next_index(self.size)

        # Some inplace collectives have custom logic for buffers and index (ReduceScatter, AllGather)
        buffer, index = self.prog.collective.get_buffer_index(self.rank, buffer, index)

        # Local copy
        if dst == self.rank:
            return self._copy(buffer, index, sendtb, ch)

        # Direct send
        assert (self.prog.topo.link(self.rank, dst)), f'No link from {self.rank} to {dst}'
        dst_chunkref = self.prog.ranks[dst].get_ref(buffer, index, self.size)
        sendOp =  Op(Instruction.send, self.rank, self, dst_chunkref, {})
        receiveOp = Op(Instruction.recv, dst, self, dst_chunkref, {})

        sendOp.prev = self.creator
        for p in sendOp.prev:
            p.next.append(sendOp)
        sendOp.next.append(receiveOp)
        receiveOp.prev.append(sendOp)

        self.prog.ranks[self.rank]._add_send(sendtb, ch, sendOp)
        self.prog.ranks[dst]._add_recv(recvtb, ch, receiveOp)
        return dst_chunkref

    def _local_reduce(self, buffer, index, tb, ch):
        dst_chunkref = self.prog.ranks[self.rank].get_ref(buffer, index, self.size)
        op = Op(Instruction.reduce, self.rank, self, dst_chunkref, {})

        op.prev = self.creator + dst_chunkref.creator
        for p in op.prev:
            p.next.append(op)

        self.prog.ranks[self.rank]._add_reduce(tb, ch, op)
        return dst_chunkref
    
    def reduce(self, dst, buffer, index=-1, sendtb=-1, recvtb=-1, ch=0):

        # Some inplace collectives have custom logic for buffers and index (ReduceScatter, AllGather)
        buffer, index = self.prog.collective.get_buffer_index(self.rank, buffer, index)

        # Local reduce
        if dst == self.rank:
            return self._local_reduce(buffer, index, sendtb, ch)

        # Receive reduce copy
        assert (self.prog.topo.link(self.rank, dst)), f'No link from {self.rank} to {dst}'
        dst_chunkref = self.prog.ranks[dst].get_ref(buffer, index, self.size)
        sendOp = Op(Instruction.send, self.rank, self, dst_chunkref, {})
        rrcOp = Op(Instruction.recv_reduce_copy, dst, self, dst_chunkref, {})

        sendOp.prev = self.creator
        for p in sendOp.prev:
            p.next.append(sendOp)
        sendOp.next.append(rrcOp)
        rrcOp.prev = dst_chunkref.creator + [sendOp]

        self.prog.ranks[self.rank]._add_send(sendtb, ch, sendOp)
        self.prog.ranks[dst]._add_receive_reduce_copy(recvtb, ch, rrcOp)
        return dst_chunkref

    def get_origin_index(self, index=0):
        return self._get_chunk(index + self.index).origin_index

    def get_origin_rank(self, index=0):
        return self._get_chunk(index + self.index).origin_rank

    def get_dst_index(self, index=0):
        return self._get_chunk(index + self.index).dst_index

    def get_dst_rank(self, index=0):
        return self._get_chunk(index + self.index).dst_rank

    def print_chunk_info(self, index=0):
        print(self._get_chunk(index + self.index)) 
