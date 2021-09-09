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
    local_ring_order = [1,3,2,6,7,5,4,0] # Reductions will happen locally within a node in this order.

    def rank(n, g):
        return local_ring_order[g] + n * num_local_gpus
        
    with SCCLProgram("allreduce_ring_dgx1", topology, 'allreduce', instances):
        s = 0

        # Chunks travels around local rings being reduced (local_gpus-1 hops) starting at local gpu 1
        # At the end of the most reduced chunk ends up on local gpu 0 every each node
        for n in range(num_nodes):
            r = rank(n, 0) # Start at local gpu 1 (index 0 in local_ring_order)
            c = Rank(r).input(0)
            for g in range(1, 8):
                next = rank(n, g)
                c = c.reduce(next, buffer=Buffer.input, index=0, step=s)
                s += 1

        # At this point gpu0 and gpu8 have the two most reduced chunks
        # 1 IB send to fully reduce chunk + 1 IB send to update other node 
        c0 = Rank(0).input(0)
        c0 = c0.send(9, buffer=Buffer.input, index=0, step=s)
        s+=1
        c1 = Rank(8).input(0)
        c1 = c1.send(1, buffer=Buffer.input, index=0, step=s)
        s += 1

        c0 = c0.reduce(8, buffer=Buffer.input, index=0, step=s) # Completely reduced chunk on node 1, gpu0
        c1 = c1.reduce(0, buffer=Buffer.input, index=0, step=s) # Completely reduced chunk on node 0, gpu0
        s += 1

        #  Propagate the fully reduced chunks
        for n in range(num_nodes):
            r = rank(n, -1) 
            c = Rank(r).input(0)
            for g in range(0, 7):
                next = rank(n, g)
                c = c.send(next, buffer=Buffer.input, index=0, step=s)
                s += 1


            

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('instances', type=int, help='number of instances')
args = parser.parse_args()

allreduce(args.num_nodes, args.instances)
