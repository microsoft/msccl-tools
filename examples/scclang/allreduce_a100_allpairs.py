# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

def allreduce_allpairs(gpus, instances, protocol):
    size = gpus
    chunksperloop = gpus * gpus
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, True)
    with SCCLProgram("allreduce_pairs", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual, dependence_nop=True):
        
        # Each rank sends the nth chunk to the nth rank into scratch space
        for r1 in range(size):
            for r2 in range(size):
                if r1 != r2:
                    index = r2 * size
                    c = chunk(r1, Buffer.input, index, size=size)
                    c.copy(r2, 'scratch', sendtb=r2, recvtb=r1)

        # Each rank performs a local reduction on the nth chunk
        # Utilize 8 threadblocks for this reduction for better parallelism
        for r in range(size):
            for index in range(0, size * (size-1)):
                    c = chunk(r, Buffer.input, r*size + (index % size))
                    c.reduce(chunk(r, 'scratch', index), sendtb=(index % size))
        
        # Each rank sends the fully reduced nth chunk to all other gpus
        for r1 in range(size):
            for r2 in range(size):
                if r1 != r2:
                    index = r1 * size
                    c = chunk(r1, Buffer.input, index, size)
                    c.copy(r2, Buffer.input, index, sendtb=r2, recvtb=r1)
                
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL128', 'LL'], help='Protocol')

args = parser.parse_args()

allreduce_allpairs(args.num_gpus, args.instances, args.protocol)