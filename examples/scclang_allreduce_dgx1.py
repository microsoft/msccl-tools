# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from sccl.language import *
from sccl.topologies.distributed import *
from sccl.topologies.nvidia import *
from sccl.collectives import *

def allreduce(num_nodes, instances):
    local_topology = dgx1()
    num_local_gpus = 8
    remote_bw = 1
    topology = distributed_fully_connected(local_topology, num_nodes, remote_bw)
    size = topology.num_nodes()
    local_ring_order = [0, 4, 6, 7, 5, 1, 3, 2] # Reductions will happen locally within a node in this order.

    def AddChunk(ib_chunks, key, c):
        if key in ib_chunks: 
            ib_chunks[key] = ib_chunks[key].group(c)
        else:
            ib_chunks[key] = c

    def rank(n, g):
        return local_ring_order[g] + n * num_local_gpus
        
    with SCCLProgram("allreduce_ring_dgx1", topology, 'allreduce', instances):
        s = 0

        # Chunks travels around local rings being reduced (local_gpus-1 hops)
        num_chunks = instances * num_local_gpus * instances
        for n in range(num_nodes):
            for g in range(num_local_gpus):
                for n2 in range(num_nodes):
                    for ch in range(instances):
                        r = rank(n, g)
                        chunk_index = rank(n2, g)*instances + ch
                        c = Rank(r).input(chunk_index)
                        next_index = (g+1) % num_local_gpus
                        next = rank(n, next_index % num_local_gpus)
                        while next != r:
                            c = c.reduce(next, buffer=Buffer.input, index=chunk_index, step=s, ch=ch)
                            next_index = (next_index + 1) % num_local_gpus
                            next = rank(n, next_index)
                            s += 1

        # Send the partly reduced chunk around the ring      
        # After this step every chunk on each node will contain the partial reduction of the local nodes chunk
        for n in range(num_nodes):
            for g in range(num_local_gpus):
                for n2 in range(num_nodes):
                    for ch in range(instances):
                        r = rank(n, g)
                        next = rank(n, (g-1)%num_local_gpus)
                        index = r*instances + ch
                        chunk_index = rank(n2, g) * instances + ch
                        c = Rank(next).input(chunk_index)
                        next_index = (g) % num_local_gpus
                        next = rank(n, next_index)
                        while next_index != (g - 1) % num_local_gpus:
                            c = c.send(next, buffer=Buffer.input, index=chunk_index, step=s, ch=ch)
                            next_index = (next_index + 1) % num_local_gpus
                            next = rank(n, next_index)
                            s += 1

        # Reduce chunks across IB with other nodes
        # Update fully reduced chunks locally
        # TODO: only works with 2 nodes
        if num_nodes > 1:
            s=1000
            c = Rank(0).input(0, size*instances)
            c = c.reduce(8, step=s, index=0, buffer=Buffer.input)
            s+=1
            c.send(0, step=s, index=0, buffer=Buffer.input)
            s +=1

            # Final round of sending the fully reduced chunks around the ring
            for n in range(num_nodes):
                for ch in range(instances):
                    g = 0
                    r = rank(n, g)
                    next = rank(n, g)
                    c = Rank(next).input(0, size*instances)
                    next_index = (g+1) % num_local_gpus
                    next = rank(n, next_index)
                    while next_index != g:
                        c = c.send(next, buffer=Buffer.input, index=0, step=s, ch=ch)
                        next_index = (next_index + 1) % num_local_gpus
                        next = rank(n, next_index)
                        s += 1

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('instances', type=int, help='number of instances')
args = parser.parse_args()

allreduce(args.num_nodes, args.instances)
