# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from enum import Enum
import functools
from msccl.language.ir import *
from msccl.language.passes import *
from msccl.language.tb_assignment import *
from msccl.language.chunk import *
from msccl.language.buffer import *
from msccl.language.rank_dag import *
import msccl.collectives as collectives
# from msccl.language.visualize import *

_current_program = None
def _curr():
    global _current_program
    if _current_program == None:
        raise RuntimeError("No Program in context")
    return _current_program

class MSCCLProgram:
    def __init__(self, name, topo, collective, instances, protocol='Simple', \
            threadblock_policy=ThreadblockPolicy.auto, interleaved_replication=True,
            instr_fusion=True, check_xml=True, dependence_nop=False):
        self.name = name
        self.topo = topo
        self.collective = collective       
        self.num_ranks = topo.num_nodes()
        self.instances = instances
        self.protocol = protocol
        self.threadblock_policy = threadblock_policy
        self.interleaved_replication = interleaved_replication
        self.instr_fusion = instr_fusion
        self.check_xml = check_xml
        self.dependence_nop = dependence_nop
        assert protocol == 'Simple' or protocol == 'LL' or protocol == 'LL128', \
            f'Given protocol: {protocol}. Must be either Simple, LL, LL128'
        self.run_opt = True # Runs optimization passes
        # Initialize the input buffers
        # self.chunk_dag = ChunkDAG()
        self.buffers = collective.init_buffers()
        self.instr_dag = InstructionDAG(self.num_ranks, self.buffers)
        for r in range(self.num_ranks):
            for index, chunk in enumerate(self.buffers[r][Buffer.input]):
                buffer, index = self.collective.get_buffer_index(r, Buffer.input, index)
                ref = self.get_ref(r, buffer, index, 1)
                # self.chunk_dag.init_chunk(chunk, ref)
                self.instr_dag.add_start(r, buffer, index, ref)

    def __enter__(self):
        global _current_program
        if _current_program != None:
            raise RuntimeError("There is already a MSCCL Program in context")
        _current_program = self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _current_program
        if _current_program != self:
            raise RuntimeError("This program is not currently in context")
        _current_program = None

    # Tracks a send operation on the buffers
    def apply_send(self, src, src_buffer, src_index, dst, dst_buffer, dst_index, size):
        src_buffer, src_index = self.collective.get_buffer_index(src, src_buffer, src_index)
        dst_buffer, dst_index = self.collective.get_buffer_index(dst, dst_buffer, dst_index)
        sb = self.buffers[src][src_buffer]
        db = self.buffers[dst][dst_buffer]
        for i in range(size):
            db[dst_index + i] = sb[src_index + i]

    # Tracks a reduce operation on the buffers
    def apply_reduce(self, src, src_buffer, src_index, dst, dst_buffer, dst_index, size):
        src_buffer, src_index = self.collective.get_buffer_index(src, src_buffer, src_index)
        dst_buffer, dst_index = self.collective.get_buffer_index(dst, dst_buffer, dst_index)
        sb = self.buffers[src][src_buffer]
        db = self.buffers[dst][dst_buffer]
        for i in range(size):
            reduce_chunk = db[dst_index + i]
            sent_chunk = sb[src_index + i]
            db[dst_index + i] = reduce_chunk.reduce(dst, sent_chunk)

    def get_ref(self, rank, buffer, index, size):
        buffer, index = self.collective.get_buffer_index(rank, buffer, index)
        return Ref(rank, buffer, index, size, self)

    def get_chunks(self, rank, buffer, index, size=1):
        chunks = [None] * size
        for i in range(0, size):
            if self.buffers[rank][buffer] and index+i < len(self.buffers[rank][buffer]):
                chunks[i] = self.buffers[rank][buffer][index+i]
            else:
                 chunks[i] = None
        return chunks

    def check_buffer_exists(self, rank, name):
        if name not in self.buffers[rank]:
            self.buffers[rank][name] = BufferSlice(Buffer.scratch, name)

    # Checks that all chunks that should be on each rank
    # are present in the output buffer.
    def check(self):
        return self.collective.check(self)

    # Lower program to XML
    def lower(self):
        # self.chunk_dag._complete_metadata()
        # self.chunk_dag.channel_assignment()
        # self.chunk_dag.lower_instr_dag(self.instr_dag)
        self.instr_dag.convert_set_list() # Pre-emptively convert sets to lists
        if self.instr_fusion:
            self.instr_dag.optimize()
        self.instr_dag._complete_metadata()
        if self.threadblock_policy == ThreadblockPolicy.manual:
            manual_assign_tbs(self.instr_dag)
        else:
            auto_assign_tbs(self.instr_dag)
        self.instr_dag.lower_pt1(self.instances)
        gpu_prgms = self.instr_dag.lower_pt2(self.instances, self.interleaved_replication)
        if self.check_xml:
            # Check generated MSCCL-IR for correctness - no circular dependencies, sends and receives are ordered
            # For very large programs, turn off check_xml when shipping 
            check_dependency_cycles(self.instr_dag.tbs)
            check_threadblock_ordering(self.instr_dag)
        return Program(self.name, self.collective.name, self.collective.inplace, self.protocol, gpu_prgms)  

    def generate_xml(self):
        return ir_to_xml(self.lower(), dependence_nop=self.dependence_nop)
    
    def print_chunk_dag(self):
        visualize_chunk_dag(self.chunk_dag.chunk_paths)

    def print_instr_dags(self, rank):
        if rank == 0:
            for r in range(len(self.ranks)):
                visualize_instr_dag(self.instr_dags[r].operations)
        else:
            visualize_instr_dag(self.instr_dags[rank].operations)

def Print():
    _curr().print_chunk_dag()

def chunk(rank, buffer, index, size=1):
    if _curr().buffers[rank][buffer][index] is None:
        return None
    return _curr().get_ref(rank, buffer, index, size)

def create_scratch(rank, name):
    return _curr().create_scratch(rank, name)

def XML():
   print(_curr().generate_xml())

def Check():
    return _curr().check()

@dataclass
class Ref(ChunkRef):
    prog: MSCCLProgram

    def __repr__(self):
        return f'Ref(Buffer:{self.buffer}, Index:{self.index}, Size:{self.size}, Rank:{self.rank})'

    def _end(self):
        return self.index + self.size

    def _get_chunk(self, index):
        return self.prog.buffers[self.rank][self.buffer][index]

    def split(self, num):
        assert (self.size % num == 0), f'Trying to split a chunk of {self.size} elements into {num} parts'
        chunks = [None] * num
        size = self.size // num
        for i in range(num):
            index = self.index + i * size
            chunks[i] = self.prog.get_ref(self.rank, self.buffer, index, size)
        return chunks

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
        return Ref(self.rank, self.buffer, first.index, end - first.index, self.prog)
        
    # Copies the chunk(s) referenced by this chunkref onto Rank dst at location (buffer, index)
    def copy(self, dst, buffer=None, index=-1, sendtb=-1, recvtb=-1, ch=-1):
        self.prog.check_buffer_exists(dst, buffer)

        # If index is not specified assume it is going to the same place in the next gpu
        if index == -1 and buffer == None:
            index = self.index
            buffer = self.buffer
        elif index == -1 and buffer is not Buffer.input and buffer is not Buffer.output:
            index = self.prog.buffers[dst][buffer].instance_size()

        # Some inplace collectives have custom logic for buffers and index (ReduceScatter, AllGather)
        buffer, index = self.prog.collective.get_buffer_index(self.rank, buffer, index)

        # Direct send
        assert (self.prog.topo.link(self.rank, dst) or dst == self.rank), f'No link from {self.rank} to {dst}'
        dst_chunkref = self.prog.get_ref(dst, buffer, index, self.size)

        # Check if we are copying the chunk to the same index (easy mistake when we are using inplace)
        if dst_chunkref == self:
            return

        # chunks = self.prog.get_chunks(self.rank, self.buffer, self.index, self.size)
        # overwritten_chunks = self.prog.get_chunks(dst, buffer, index, self.size)
        
        self.prog.apply_send(self.rank, self.buffer, self.index, dst, buffer, index, self.size)

        # self.prog.chunk_dag.add_send(chunks, overwritten_chunks, self, dst_chunkref, sendtb, recvtb, ch)
        sender = self.rank
        receiver = dst
        if sender != receiver:
            sop = self.prog.instr_dag.add_send(sender, self, dst_chunkref, sendtb, ch)
            rop = self.prog.instr_dag.add_recv(receiver, self, dst_chunkref, recvtb, ch, sop)
            sop.recv_match = rop
        else:
            self.prog.instr_dag.add_copy(sender, self, dst_chunkref, sendtb, ch)

        return dst_chunkref

    # Reduces the chunk(s) referenced by other_chunkref into the chunk(s) referenced by this chunkref
    def reduce(self, other_chunkref, sendtb=-1, recvtb=-1, ch=-1):
        # Receive reduce copy
        dst = self.rank
        src = other_chunkref.rank
        assert (self.prog.topo.link(src, dst) or src == dst), f'No link from {src} to {dst}'
        # dst_chunkref = self.prog.get_ref(dst, buffer, index, self.size)

        # chunks1 = self.prog.get_chunks(self.rank, self.buffer, self.index, self.size)
        # chunks2 = self.prog.get_chunks(other_chunkref.rank, other_chunkref.buffer, other_chunkref.index self.size)

        self.prog.apply_reduce(src, other_chunkref.buffer, other_chunkref.index, dst, self.buffer, self.index, self.size)

        # reduce_chunks = self.prog.get_chunks(dst, buffer, index, self.size)
        # self.prog.chunk_dag.add_reduce(chunks1, chunks2, reduce_chunks, self, dst_chunkref, sendtb, recvtb, ch)
        if src != dst:
            sop = self.prog.instr_dag.add_send(src, other_chunkref, self, sendtb, ch)
            rop = self.prog.instr_dag.add_recv_reduce_copy(dst, other_chunkref, self, recvtb, ch, sop)
            sop.recv_match = rop
        else:
            self.prog.instr_dag.add_reduce(src, other_chunkref, self, sendtb, ch)

        return self

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


# @dataclass
# class ChunkOp():
#     inst: ChunkInstruction
#     src: Ref # Ref Chunk acted on
#     dst: Ref # Ref Chunk created
#     sendtb: int = -1# For lowering to RankInstructions
#     recvtb: int = -1#  For lowering to RankInstructions
#     ch: int = -1 # For lowering to RankInstructions
#     steps_from_start:int  = -1
#     steps_to_end: int = -1 
#     prev: list = field(default_factory=list) # Previous ChunkOps
#     next: list = field(default_factory=list) # Next ChunkOps
#     visited = False
#     num = -1

#     def __repr__(self):
#         return f'ChunkOp({self.inst} {self.dst.rank} {self.dst.buffer} {self.dst.index})'

#     def __lt__(self, other):
#         return self.steps_from_start < other.steps_from_start

#     def __hash__(self):
#         return hash((self.inst, self.dst.rank, self.dst.index, self.dst.buffer)) # TODO 

# def same_slot(ref1, ref2):
#     return ref1.rank == ref2.rank and ref1.buffer == ref2.buffer and ref1.index == ref2.index

# # Returns if there is overlap between the refs
# def overlap_refs(ref1, ref2):
#     same_location = ref1.rank == ref2.rank and ref1.buffer == ref2.buffer
#     if same_location:
#         ref1_range = (ref1.index, ref1.index + ref1.size)
#         ref2_range = (ref2.index, ref2.index + ref2.size)
#         if ref1_range < ref2_range:
#             return ref1_range[0] < ref2_range[1]
#         else:
#             return ref2_range[0] < ref1_range[1]
#     return False

# class ChunkDAG:

#     def __init__(self):
#         self.chunks = []
#         self.chunk_paths = {} # chunk -> ChunkOp. Stores the entry point to where every chunk is created
#         self.max_hops = -1

#     # Initialize the ChunkDAG with starting chunks
#     def init_chunk(self, chunk, ref):
#         op = ChunkOp(ChunkInstruction.start, None, ref, steps_from_start=-1)
#         self.chunks.append(chunk)
#         self.chunk_paths[chunk] = op

#     def _find_prev_op_for_chunk(self, chunk, ref):
#         prev_op = None
#         frontier = [self.chunk_paths[chunk]]
#         while len(frontier) > 0:
#             current_op = frontier[0]
#             if overlap_refs(ref, current_op.dst):
#                 prev_op = current_op
#             frontier = frontier[1:] + current_op.next
#         return prev_op

#     def add_send(self, chunks, overwritten_chunks, src, dst, sendtb, recvtb, ch):
#         # Find the previous operation for these chunks
#         prev_ops = []
#         steps_from_start = 0
#         for chunk1, chunk2 in zip(chunks, overwritten_chunks):
#             prev_op_src = self._find_prev_op_for_chunk(chunk1, src)
#             if chunk2 is None:
#                 steps_from_start = max(steps_from_start, prev_op_src.steps_from_start)
#             else:
#                 prev_op_dst = self._find_prev_op_for_chunk(chunk2, dst) # In case we overwrite
#                 steps_from_start = max(prev_op_src.steps_from_start, prev_op_dst.steps_from_start, steps_from_start)
#                 prev_ops.append(prev_op_dst)
#             prev_ops.append(prev_op_src)
#             # prev_op = self._find_prev_op_for_chunk(chunk, src)
#             # steps_from_start = max(steps_from_start, prev_op.steps_from_start)
#             # prev_ops.append(prev_op)
#         op = ChunkOp(ChunkInstruction.send, src, dst, sendtb, recvtb, ch, steps_from_start+1)
        
#         for prev_op in prev_ops:
#             prev_op.next.append(op)
#         op.prev = prev_ops

#     def add_reduce(self, chunks1, chunks2, reduce_chunks, src, dst, sendtb, recvtb, ch):
#         # self.chunks.append(reduce_chunks)
#         prev_ops = []
#         steps_from_start = 0
#         # Find the previous operations that reduce builds off
#         for chunk1, chunk2 in zip(chunks1, chunks2):
#             prev_op_src = self._find_prev_op_for_chunk(chunk1, src)
#             prev_op_dst = self._find_prev_op_for_chunk(chunk2, dst)
#             steps_from_start = max(prev_op_src.steps_from_start, prev_op_dst.steps_from_start, steps_from_start)
#             prev_ops.append(prev_op_src)
#             prev_ops.append(prev_op_dst)
            
#         op = ChunkOp(ChunkInstruction.reduce, src, dst, sendtb, recvtb, ch, steps_from_start+1)

#         for prev_op in prev_ops:
#             prev_op.next.append(op)
#             op.prev.append(prev_op)

#         # Reduce operations create new chunks, so keep a pointer to a new chunk
#         for rc in reduce_chunks:
#             self.chunk_paths[rc] = op

#     def _complete_metadata(self):
#         def dfs(op):
#             if len(op.next) == 0:
#                 op.steps_to_end = 0
#             else:
#                 for o in op.next:
#                     dfs(o)
#                 op.steps_to_end = functools.reduce(lambda cur, x: max(cur, x.steps_to_end+1), op.next, 0)

#         for chunk, op in self.chunk_paths.items():
#             if op.inst == ChunkInstruction.start:
#                 dfs(op)
            

#     # Assigns each send and a reduce a channel for communication based of policies
#     def channel_assignment(self, channel_policy='zero'):
#         frontier = []
#         visited = set()
#         for chunk, op in self.chunk_paths.items():
#             if len(op.prev) == 0: 
#                 heapq.heappush(frontier, op)

#         # If an op isn't annotated with a channel set it to 0
#         if channel_policy == 'zero':
#             while len(frontier) > 0:
#                 op = heapq.heappop(frontier)
#                 if op not in visited:
#                     op.ch = 0 if op.ch == -1 else op.ch
#                     for o in op.next:
#                         heapq.heappush(frontier, o)
#                     visited.add(op)

#     def lower_instr_dag(self, instr_dag):
#         frontier = []
#         visited = set()

#         for chunk, op in self.chunk_paths.items():
#             if len(op.prev) == 0: 
#                 heapq.heappush(frontier, ((op.steps_from_start, op.steps_to_end), op))

#         while len(frontier) > 0:
#             _, op = heapq.heappop(frontier)
#             if op not in visited:
#                 sendtb = op.sendtb
#                 recvtb = op.recvtb
#                 ch =  op.ch
#                 if op.inst == ChunkInstruction.start:
#                     rank = op.dst.rank
#                     instr_dag.add_start(rank, op.dst.buffer, op.dst.index, op.dst)
#                 elif op.inst == ChunkInstruction.send:
#                     sender = op.src.rank
#                     receiver = op.dst.rank
#                     if sender != receiver:
#                         sop = instr_dag.add_send(sender, op.src, op.dst, op.steps_from_start*2, op.steps_to_end*2+1, sendtb, ch)
#                         rop = instr_dag.add_recv(receiver, op.src, op.dst, op.steps_from_start*2+1, op.steps_to_end*2, recvtb, ch)
#                         sop.match = [rop]
#                     else:
#                         instr_dag.add_copy(sender, op.src, op.dst, op.steps_from_start*2, op.steps_to_end*2, sendtb, ch)
#                 elif op.inst == ChunkInstruction.reduce:
#                     sender = op.src.rank
#                     receiver = op.dst.rank
#                     if sender != receiver:
#                         sop = instr_dag.add_send(sender, op.src, op.dst, op.steps_from_start*2,op.steps_to_end*2+1, sendtb, ch)
#                         rop = instr_dag.add_recv_reduce_copy(receiver, op.src, op.dst, op.steps_from_start*2+1, op.steps_to_end*2, recvtb, ch)
#                         sop.match = [rop]
#                     else:
#                         instr_dag.add_reduce(sender, op.src, op.dst, op.steps_from_start*2, op.steps_to_end*2, sendtb, ch)

#                 for o in op.next:
#                     heapq.heappush(frontier, ((o.steps_from_start, o.steps_to_end), o))
#                 visited.add(op)
#         instr_dag.convert_set_list() # Pre-emptively convert sets to lists
