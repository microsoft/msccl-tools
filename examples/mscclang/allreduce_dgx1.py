# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from msccl.language import *
from msccl.topologies.distributed import *
from msccl.topologies.nvidia import *
from msccl.language.collectives import AllReduce

def allreduce(num_nodes, instances):
    local_topology = dgx1()
    num_local_gpus = 8
    remote_bw = 1
    topology = distributed_fully_connected(local_topology, num_nodes, remote_bw)
    size = topology.num_nodes()
    collective = AllReduce(size, 1, True)
    local_ring_order = [1,3,2,6,7,5,4,0]

    def rank(n, g):
        return local_ring_order[g] + n * num_local_gpus
        
    with MSCCLProgram("allreduce_ring_dgx1", topology, collective, 1):

        # Chunks travels around local rings being reduced (local_gpus-1 hops) starting at local gpu 1
        # At the end of the most reduced chunk ends up on local gpu 0 every each node
        for n in range(num_nodes):
            r = rank(n, 0) # Start at local gpu 1 (index 0 in local_ring_order)
            c = chunk(r, Buffer.input, 0)
            for g in range(1, 8):
                c = c.reduce(rank(n,g), Buffer.input, 0)
    
        # At this point gpu0 and gpu8 have the two most reduced chunks
        # 1 IB send to fully reduce chunk + 1 IB send to update other node 

        chunk(0, Buffer.input, 0).send(9, Buffer.input, 0)
        chunk(8, Buffer.input, 0).send(1, Buffer.input, 0).reduce(0, Buffer.input, 0)
        chunk(9, Buffer.input, 0).reduce(8, Buffer.input, 0)

        #  Propagate the fully reduced chunks going backwards around the ring
        for n in range(num_nodes):
            r = rank(n, 7) 
            c = chunk(r, Buffer.input, 0)
            for g in range(6, -1, -1):
                next = rank(n, g)
                c = c.send(next, Buffer.input, 0)

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('instances', type=int, help='number of instances')
args = parser.parse_args()

assert args.num_nodes == 2, "Only works for 2 nodes right now"

allreduce(args.num_nodes, args.instances)
