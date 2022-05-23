from dataclasses import dataclass, field
from msccl.language.ir import Buffer
from msccl.language import *

class Collective():
    def __init__(self, num_ranks, chunk_factor, inplace):
        self.num_ranks = num_ranks
        self.chunk_factor = chunk_factor
        self.inplace = inplace
        self.name = "custom"

    def init_buffers(self):
        pass

    def check(self, prog):
        pass

    def get_buffer_index(self, rank, buffer, index):
        return buffer, index


class AllToAll(Collective):

    def __init__(self, num_ranks, chunk_factor, inplace):
        Collective.__init__(self, num_ranks, chunk_factor, inplace)
        self.name = 'alltoall'

    def init_buffers(self):
        chunks_per_node = self.num_ranks * self.chunk_factor
        rank_buffers = []
        for r in range(self.num_ranks):
            input_buffer = [None] * chunks_per_node
            output_buffer = [None] * chunks_per_node
            for index in range(chunks_per_node):
                chunk = Chunk(r, index, index//self.chunk_factor, index % self.chunk_factor + r*self.chunk_factor)
                input_buffer[index] = chunk
            if self.inplace:
                buffers = {Buffer.input : input_buffer, 
                    Buffer.output : input_buffer}
            else:
                buffers = {Buffer.input : input_buffer, 
                        Buffer.output : output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers

    # Expected output buffer for alltoall
    def check(self, prog):
        chunks_per_node = self.num_ranks * self.chunk_factor
        correct = True
        for r in range(self.num_ranks):
            output = prog.buffers[r][Buffer.output]
            for i in range(self.num_ranks):
                for ch in range(self.chunk_factor):
                    index = ch + i * self.chunk_factor
                    chunk = output[index]
                    expected_origin_index = ch + r * self.chunk_factor
                    if chunk is None or chunk.origin_rank != i or chunk.origin_index != expected_origin_index:
                        print(f'Rank {r} chunk {index} is incorrect should be chunk({i},{expected_origin_index}) given {chunk}')
                        correct = False
        return correct


class AllGather(Collective):
    def __init__(self, num_ranks, chunk_factor, inplace):
        Collective.__init__(self, num_ranks, chunk_factor, inplace)
        self.name = 'allgather'

    # Initializes input buffer for an allgather
    def init_buffers(self):
        rank_buffers = []
        if self.inplace:
            # Inplace AllGather only uses the output buffer   
            for r in range(self.num_ranks):
                output_buffer = [None] * (self.num_ranks * self.chunk_factor)
                for ch in range(self.chunk_factor):
                    output_buffer[r*self.chunk_factor+ch] = Chunk(r, ch, -1, r*self.chunk_factor+ch)
                buffers = {Buffer.input : output_buffer[r*self.chunk_factor:(r+1)*self.chunk_factor],
                           Buffer.output : output_buffer}
                rank_buffers.append(buffers)
        else:
            for r in range(self.num_ranks):
                input_buffer = [None] * self.chunk_factor
                output_buffer = [None] * (self.num_ranks * self.chunk_factor)
                for ch in range(self.chunk_factor):
                    input_buffer[ch] = Chunk(r, ch, -1, r*self.chunk_factor+ch)
                buffers = {Buffer.input : input_buffer, 
                           Buffer.output : output_buffer}
                rank_buffers.append(buffers)
        return rank_buffers
                
    # Expected output buffer for allgather
    def check(self, prog):
        correct = True
        buf = Buffer.output
        for r in range(self.num_ranks):
            output = prog.buffers[r][buf]
            for i in range(self.num_ranks):
                for ch in range(self.chunk_factor):
                    index = i*self.chunk_factor + ch
                    chunk = output[index]
                    if chunk is None:
                        print(f'Rank {r} chunk {index} is incorrect should be ({i}, {ch}) given None')
                        correct = False
                    elif chunk.origin_rank != i or chunk.origin_index != ch:
                        print(f'Rank {r} chunk {index} is incorrect should be ({i}, {ch}) given ({chunk.origin_rank}, {chunk.origin_index})')
                        correct = False
        return correct

    
    def get_buffer_index(self, rank, buffer, index):
        # For inplace AllGathers, the input buffer points into the output buffer
        if self.inplace and buffer == Buffer.input:
            return Buffer.output, index + rank * self.chunk_factor
        else:
            return buffer, index


            
class AllReduce(Collective):

    def __init__(self, num_ranks, chunk_factor, inplace):
        Collective.__init__(self, num_ranks, chunk_factor, inplace)
        self.name = 'allreduce'

    def init_buffers(self):
        chunks_per_node = self.chunk_factor
        rank_buffers = []
        for r in range(self.num_ranks):
            input_buffer = []
            output_buffer = [None] * chunks_per_node
            for c in range(chunks_per_node):
                # Chunks start at rank r index c, and ends on all ranks (-1) at index r
                input_buffer.append(Chunk(r, c, -1, c))
            # Input and output buffer are the same.
            if self.inplace:
                buffers = {Buffer.input : input_buffer, 
                           Buffer.output : input_buffer}
            else:
                buffers = {Buffer.input : input_buffer, 
                           Buffer.output : output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers

    def check(self, prog):
        chunks_per_node = self.chunk_factor
        expected_chunks = []
        buf = Buffer.input if self.inplace else Buffer.output

        for c in range(chunks_per_node):
            chunk = ReduceChunk(-1, [])
            for r in range(self.num_ranks):
                chunk = chunk.reduce(-1, Chunk(r, c))
            expected_chunks.append(chunk)

        correct = True
        for r in range(self.num_ranks):
            output = prog.buffers[r][buf]
            for c in range(chunks_per_node):
                chunk = output[c]
                if chunk is None or chunk != expected_chunks[c]:
                    print(f'Rank {r} chunk {c} is incorrect should be ReduceChunk index {c} from all ranks, given {chunk}')
                    correct = False
        return correct

    def get_buffer_index(self, rank, buffer, index):
        if self.inplace and buffer == Buffer.output:
            return Buffer.input, index
        else:
            return buffer, index


class ReduceScatter(Collective):
    def __init__(self, num_ranks, chunk_factor, inplace):
        Collective.__init__(self, num_ranks, chunk_factor, inplace)
        self.name = 'reduce_scatter'

    def init_buffers(self):
        rank_buffers = []
        for r in range(self.num_ranks):
            if self.inplace:
                input_buffer = []
                for i in range(self.num_ranks):
                    for c in range(self.chunk_factor):
                        input_buffer.append(Chunk(r, i*self.chunk_factor + c, i, c))
                buffers = {Buffer.input : input_buffer}
                rank_buffers.append(buffers)
            else:
                input_buffer = []
                output_buffer = [None] * self.chunk_factor
                for i in range(self.num_ranks):
                    for c in range(self.chunk_factor):
                        input_buffer.append(Chunk(r, i*self.chunk_factor + c, i, c))
                buffers = {Buffer.input : input_buffer, 
                        Buffer.output : output_buffer}
                rank_buffers.append(buffers)
        return rank_buffers

    def check(self, prog):
        expected_chunks = []
        buf = Buffer.input if self.inplace else Buffer.output
        for c in range(self.num_ranks * self.chunk_factor):
            chunk = ReduceChunk(-1, [])
            for r in range(self.num_ranks):
                chunk = chunk.reduce(-1, Chunk(r, c))
            expected_chunks.append(chunk)

        correct = True
        for r in range(self.num_ranks):
            output = prog.buffers[r][buf]
            for c in range(self.chunk_factor):
                correct_idx = r * self.chunk_factor + c
                if self.inplace:
                    c = correct_idx
                chunk = output[c]
                if chunk is None or chunk != expected_chunks[correct_idx]:
                    print(f'Rank {r} chunk {c} is incorrect should be index {correct_idx} from all ranks given {chunk}')
                    correct = False
        return correct

    def get_buffer_index(self, rank, buffer, index):
        # For inplace ReduceScatter the output buffer is a pointer into the input buffer
        if self.inplace and buffer == Buffer.output:
            return Buffer.input, index + rank * self.chunk_factor
        else:
            return buffer, index

