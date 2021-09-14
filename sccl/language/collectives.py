from dataclasses import dataclass, field
from sccl.language.ir import Buffer
from sccl.language import *

@dataclass
class Collective():
    num_ranks: int 
    instances: int
    inplace: bool

    def init_buffers(self):
        pass

    def check(self, prog):
        pass


class AllToAll(Collective):

    def init_buffers(self):
        chunks_per_node = self.num_ranks * self.instances
        rank_buffers = []
        for r in range(self.num_ranks):
            input_buffer = [None] * chunks_per_node
            output_buffer = [None] * chunks_per_node
            for index in range(chunks_per_node):
                chunk = Chunk(r, index, index//self.instances, index % self.instances + r*self.instances)
                input_buffer[index] = chunk
            buffers = {Buffer.input : input_buffer, 
                    Buffer.output : output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers

    # Expected output buffer for alltoall
    def check(self, prog):
        chunks_per_node = self.num_ranks * self.instances
        correct = True
        for r in range(self.num_ranks):
            output = prog.ranks[r].buffers[Buffer.output]
            for i in range(self.num_ranks):
                for ch in range(self.instances):
                    index = ch + i * self.instances
                    chunk = output[index]
                    expected_origin_index = ch + r * self.instances
                    if chunk is None or chunk.origin_rank != i or chunk.origin_index != expected_origin_index:
                        print(f'Rank {r} chunk {index} is incorrect should be chunk({i},{expected_origin_index}) given {chunk}')
                        correct = False
        return correct


class AllGather(Collective):
    # Initializes input buffer for an allgather
    def init_buffers(self):
        rank_buffers = []
        for r in range(self.num_ranks):
            input_buffer = []
            output_buffer = [None] * (self.num_ranks * self.instances)
            for ch in range(self.instances):
                input_buffer.append(Chunk(r, ch, -1, r*self.instances+ch))
            buffers = {Buffer.input : input_buffer, 
                    Buffer.output : output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers
                
    # Expected output buffer for allgather
    def check(self, prog):
        correct = True
        buf = Buffer.input if self.inplace else Buffer.output
        for r in range(self.num_ranks):
            output = prog.ranks[r].buffers[buf]
            for i in range(self.num_ranks):
                for ch in range(self.instances):
                    index = i*self.instances + ch
                    chunk = output[index]
                    if chunk is None or chunk.origin_rank != i or chunk.origin_index != ch:
                        print(f'Rank {r} chunk {index} is incorrect should be ({i}, {ch}) given {chunk}')
                        correct = False
        return correct

            
class AllReduce(Collective):

    def init_buffers(self):
        chunks_per_node = self.instances
        rank_buffers = []
        for r in range(self.num_ranks):
            input_buffer = []
            output_buffer = [None] * chunks_per_node
            for c in range(chunks_per_node):
                # TODO: Chunk starts at rank r index c, and ends on all ranks (-1) at index r, also reduced (?? how to indicate??)
                input_buffer.append(Chunk(r, c, -1, c))
            buffers = {Buffer.input : input_buffer, 
                    Buffer.output : output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers

    def check(self, prog):
        chunks_per_node = self.instances
        expected_chunks = []

        for c in range(chunks_per_node):
            chunk = ReduceChunk([])
            for r in range(self.num_ranks):
                chunk = chunk.reduce(Chunk(r, c))
            expected_chunks.append(chunk)

        correct = True
        for r in range(self.num_ranks):
            output = prog.ranks[r].buffers[Buffer.input]
            for c in range(chunks_per_node):
                chunk = output[c]
                if chunk is None or chunk != expected_chunks[c]:
                    print(f'Rank {r} chunk {c} is incorrect should be {expected_chunks[c]} given {chunk}')
                    correct = False
        return correct


class ReduceScatter(Collective):

    def init_buffers(self):
        rank_buffers = []
        for r in range(self.num_ranks):
            input_buffer = []
            output_buffer = [None] * self.instances
            for i in range(self.num_ranks):
                for c in range(self.instances):
                    input_buffer.append(Chunk(r, i*self.instances + c, i, c))
            buffers = {Buffer.input : input_buffer, 
                    Buffer.output : output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers

    def check(self, prog):
        expected_chunks = []
        buf = Buffer.input if self.inplace else Buffer.output

        for c in range(self.num_ranks * self.instances):
            chunk = ReduceChunk([])
            for r in range(self.num_ranks):
                chunk = chunk.reduce(Chunk(r, c))
            expected_chunks.append(chunk)

        correct = True
        for r in range(self.num_ranks):
            output = prog.ranks[r].buffers[buf]
            for c in range(self.instances):
                chunk = output[c]
                correct_idx = r * self.instances + c
                if chunk is None or chunk != expected_chunks[correct_idx]:
                    print(f'Rank {r} chunk {c} is incorrect should be {expected_chunks[correct_idx]} given {chunk}')
                    correct = False
        return correct
