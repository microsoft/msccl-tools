# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from msccl.language import *
from msccl.topologies.distributed import *
from msccl.topologies import *
from msccl.language.collectives import Collective

class Pipeline(Collective):
    def init_buffers(self):
        chunks_per_node = self.chunk_factor
        rank_buffers = []
        for r in range(self.num_ranks):
            input_buffer = [None] * chunks_per_node
            output_buffer = [None] * chunks_per_node
            if r != 0:
                for c in range(chunks_per_node):
                    input_buffer[c] = Chunk(r, c, r-1, c)
            buffers = {Buffer.input : input_buffer, 
                       Buffer.output : output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers
            

    # Final state chunks on rank(i) end up on rank(i-1)
    def check(self, prog):
        correct = True
        for r in range(0, self.num_ranks-1):
            output = prog.buffers[r][Buffer.output]
            for c in range(self.chunk_factor):
                chunk = output[c]
                if chunk is None or chunk.origin_rank != r+1 or chunk.origin_index != c:
                    print(f'Rank {r} chunk {c} is incorrect should be ({r+1}, {c}) given {chunk}')
                    correct = False
        return correct


def pipeline(num_nodes, instances):
    num_local_gpus = 8
    chunks = num_local_gpus
    chunk_factor = chunks
    remote_bw = 1
    size = num_local_gpus * num_nodes
    topology = fully_connected(size)
    collective = Pipeline(size, chunk_factor, False)

    def rank(node, local_rank):
        return node * num_local_gpus + local_rank
    
    with MSCCLProgram("alltonext-backwards", topology, collective, instances):

        for n in range(num_nodes):
            for g in range(num_local_gpus):
                r = rank(n, g)

                # Do nothing for first gpu - end of pipeline
                if r == 0:
                    continue

                # Cross node copy - cooperative
                if g == 0:
                    for ch in range(chunks):
                        c = chunk(r, Buffer.input, ch)
                        if ch == 0:
                            # 2 steps: IB copy to (node-1, g) then gather onto (node+1, num_local_gpus-1)
                            c = c.copy(rank(n-1, ch), f's{n}->{n+1}', 0, ch=ch%2)
                        elif ch == num_local_gpus-1: 
                            # 2 steps: Scatter - copy to (node, num_local_gpus-1), IB copy to (node+1, num_local_gpus-1)
                            c = c.copy(rank(n, ch), f's{n}->{n+1}', 0, ch=ch%2)
                        else:
                            # 3 steps: Scatter - copy to (node, g), IB copy to (node-1, g), gather onto (node-1, num_local_gpus-1)
                            c = c.copy(rank(n, ch), f's{n}->{n+1}', 0, ch=ch%2)
                            c = c.copy(rank(n-1, ch), f's{n}->{n+1}', 0, ch=ch%2)
                        c.copy(r-1, Buffer.output, c.get_dst_index(), ch=ch%2)
                        
                # Normal copy - directly
                else:
                    c = chunk(r, Buffer.input, 0, chunks)
                    c.copy(r-1, Buffer.output, 0, ch=g%2)
        
        Check()
        XML()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_nodes', type=int, help ='number of nodes')
    parser.add_argument('instances', type=int, help ='number of instances')

    args = parser.parse_args()

    pipeline(args.num_nodes, args.instances)