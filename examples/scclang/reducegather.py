# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import Collective

class ReduceGather(Collective):
    def __init__(self, num_ranks, chunk_factor, inplace, groups):
        Collective.__init__(self, num_ranks, chunk_factor, inplace)
        self.groups = groups
        self.gpus_per_group = num_ranks // groups
        assert chunk_factor == 1, "Only supports chunks == number of ranks"

    def init_buffers(self):
        assert self.chunk_factor == 1
        rank_buffers = []
        chunks_per_node = self.num_ranks
        if self.inplace:
            # Only an output buffer for inplace
            for r in range(self.num_ranks):
                output_buffer = [None] * chunks_per_node
                group = r // self.gpus_per_group
                for c in range(self.gpus_per_group):
                    output_buffer[group*self.gpus_per_group + c] = Chunk(r, c, -1, self.gpus_per_group*group+c)
                input_buffer = output_buffer[group*self.gpus_per_group:(group+1)*self.gpus_per_group]
                buffers = {Buffer.input: input_buffer,
                    Buffer.output : output_buffer}
                rank_buffers.append(buffers)
        else:
            for r in range(self.num_ranks):
                input_buffer = [None] * self.gpus_per_group
                output_buffer = [None] * chunks_per_node
                for c in range(self.gpus_per_group):
                    input_buffer[c] = Chunk(r, c, -1, c)
                buffers = {Buffer.input : input_buffer, 
                        Buffer.output : output_buffer}
                rank_buffers.append(buffers)
        return rank_buffers
            

    def check(self, prog):
        expected_chunks = []
        for r in range(self.num_ranks):
            chunk = ReduceChunk([])
            for x in range(self.gpus_per_group):
                y = r // self.groups
                next = y * self.groups + x
                chunk = chunk.reduce(Chunk(next, r % self.gpus_per_group))
            expected_chunks.append(chunk)

        correct = True
        for r in range(self.num_ranks):
            output = prog.buffers[r][Buffer.output]
            for c in range(self.num_ranks):
                chunk = output[c]
                if chunk is None or chunk != expected_chunks[c]:
                    print(f'Rank {r} chunk {c} is incorrect should be {expected_chunks[c]} given {chunk}')
                    correct = False
        return correct

    # For inplace Reduce Gather the input buffer is pointer into the output buffer
    def get_buffer_index(self, rank, buffer, index):
        if self.inplace and buffer == Buffer.input:
            group = rank // self.gpus_per_group
            return Buffer.output, index + group * self.gpus_per_group
        else:
            return buffer, index


# Inplace reduce gather
def reduce_gather_ip(num_ranks, groups, instances, protocol):
    gpus_per_group = num_ranks // groups
    topology = fully_connected(num_ranks)
    chunk_factor = 1
    inplace = True
    collective = ReduceGather(num_ranks, chunk_factor, inplace, groups)

    with SCCLProgram("reduce-gather", topology, collective, instances, protocol, threadblock_policy=ThreadblockPolicy.auto):

        # Per group reduce scatter
        for y in range(groups):
            for x in range(gpus_per_group):
                output_index = y * groups + x
                input_index = x
                gpu = y * groups + (x+1) % gpus_per_group
                c = chunk(gpu, Buffer.output, output_index)
                # Use the input buffer to perform reduction across groups
                for x_ in range(1, gpus_per_group):
                    c = c.reduce(y * groups + (x + 1 + x_) % gpus_per_group, Buffer.input, input_index)

        # Ring Allgather
        for r in range(num_ranks):
            c = chunk(r, Buffer.output, r)
            next = (r + 1) % num_ranks
            while next != r:
                c = c.send(next, Buffer.output, r)
                next = (next + 1) % num_ranks

        Check()
        XML()

# Out of place reduce gather 
def reduce_gather_oop(num_ranks, groups, instances, protocol):
    gpus_per_group = num_ranks // groups
    topology = fully_connected(num_ranks)
    chunk_factor = 1
    inplace = False
    collective = ReduceGather(num_ranks, chunk_factor, inplace, groups)

    with SCCLProgram("reduce-gather", topology, collective, instances, protocol, threadblock_policy=ThreadblockPolicy.auto):

        # Per group reduce scatter
        for y in range(groups):
            for x in range(gpus_per_group):
                output_index = y * groups + x
                input_index = x
                gpu = y * groups + (x+1) % gpus_per_group
                c = chunk(gpu, Buffer.input, input_index)
                # Use the input buffer to perform reduction across groups
                for x_ in range(1, gpus_per_group):
                    c = c.reduce(y * groups + (x + 1 + x_) % gpus_per_group, Buffer.input, input_index)
                # Copy reduced chunk into the output buffer
                c = c.send(c.rank, Buffer.output, output_index)


        # Ring Allgather
        for r in range(num_ranks):
            c = chunk(r, Buffer.output, r)
            next = (r + 1) % num_ranks
            while next != r:
                c = c.send(next, Buffer.output, r)
                next = (next + 1) % num_ranks

        Check()
        XML()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_ranks', type=int, help ='number of ranks')
    parser.add_argument('groups', type=int, help='number of reduction groups')
    parser.add_argument('--instances', type=int, default=1, help='number of instances')
    parser.add_argument('--protocol', type=str, default='Simple', 
        choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol')
    parser.add_argument('--outofplace', action='store_true')
    args = parser.parse_args()

    assert args.num_ranks % args.groups == 0

    if args.outofplace:
        reduce_gather_oop(args.num_ranks, args.groups, args.instances, args.protocol)
    else:
        reduce_gather_ip(args.num_ranks, args.groups, args.instances, args.protocol)
