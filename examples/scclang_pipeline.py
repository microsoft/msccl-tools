# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from sccl.language import *
from sccl.topologies.distributed import *
from sccl.topologies import *
from sccl.language.collectives import Collective

class Pipeline(Collective):
    def init_buffers(self):
        chunks_per_node = self.instances
        rank_buffers = []
        for r in range(self.num_ranks):
            input_buffer = [None] * chunks_per_node
            output_buffer = [None] * chunks_per_node
            if r != self.num_ranks -1:
                for c in range(chunks_per_node):
                    input_buffer[c] = Chunk(r, c, r+1, c)
            buffers = {Buffer.input : input_buffer, 
                       Buffer.output : output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers
            

    # Final state chunks on rank(i) end up on rank(i+1)
    def check(self, prog):
        correct = True
        for r in range(1, self.num_ranks):
            output = prog.ranks[r].buffers[Buffer.output]
            for c in range(self.instances):
                chunk = output[c]
                if chunk is None or chunk.origin_rank != r-1 or chunk.origin_index != c:
                    print(f'Rank {r} chunk {c} is incorrect should be ({r-1}, {c}) given {chunk}')
                    correct = False
        return correct


def pipeline(num_nodes):
    local_topology = dgx1()
    num_local_gpus = 8
    instances = num_local_gpus
    remote_bw = 1
    size = num_local_gpus * num_nodes
    topology = fully_connected(size)
    collective = Pipeline(size, instances, False, "pipeline")

    def rank(node, local_rank):
        return node * num_local_gpus + local_rank
    
    with SCCLProgram("pipeline", topology, collective, instances):

        # Allocate scratch space
        for n in range(num_nodes):
            for g in range(num_local_gpus):
                r1 = rank(n, g)
                if n < num_nodes: # Gather scratch
                    Rank(r1).create_scratch('gather') 
                if n > 0: # Scatter scratch
                    Rank(r1).create_scratch('scatter') 


        for n in range(num_nodes):
            for g in range(num_local_gpus):
                r = rank(n, g)

                # Do nothing for last gpu - end of pipeline
                if r == size - 1:
                    continue

                # Cross node send
                if g == num_local_gpus -1:
                    for ch in range(instances):
                        c = Rank(num_local_gpus-1).input(ch)
                        if ch == 0:
                            c = c.send(rank(n, ch), 'gather', 0, sendtb=ch, recvtb=0, ch=ch%2)

                        elif ch == num_local_gpus-1:
                            c = c.send(rank(n+1, ch), 'scatter', 0, sendtb=ch, recvtb=0, ch=ch%2)
                        else:
                            c = c.send(rank(n, ch), 'gather', 0, sendtb=ch, recvtb=0, ch=ch%2)
                            c = c.send(rank(n+1, ch), 'scatter', 0, sendtb=0, recvtb=0, ch=ch%2)
                        
                        c.send(r+1, Buffer.output, c.get_dst_index(), sendtb=0, recvtb=ch, ch=ch%2)
                        
                # Normal send
                else:
                    for ch in range(instances):
                        c = Rank(r).input(ch)
                        c.send(r+1, Buffer.output, ch, sendtb=10, recvtb=10, ch=2)
        
        Check()
        XML()
parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int, help ='number of nodes')
args = parser.parse_args()

assert args.num_nodes == 2, "Currently only working for two nodes"
pipeline(args.num_nodes)