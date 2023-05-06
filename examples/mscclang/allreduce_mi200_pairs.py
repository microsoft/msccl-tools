# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def allreduce_ltd_pairs(gpus, instances, protocol):
    # Define ranks/GPUs those would be performing reduction in the system
    # For max hops=3, one or more inner set of GPU can be selected
    # 1, 2, 9, 10
    gpuIds = [1, 2, 9, 10]      # GPU IDs that perform reduction, max hops =3

    # For max hops=4, following set of GPUs can be selected
    # 0, 1, 2, 3, 8, 9, 10, 11
    rsize = len(gpuIds)     # number of reducer ranks in system, also, number of chunks per rank
    size = 16        # chunks multiplier
    chunksperloop = size * rsize # Total number of chunks per rank
    topology = fully_connected(gpus)
    collective = AllReduce(gpus, chunksperloop, True)

    with MSCCLProgram("allreduce_ltd_pairs", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual, dependence_nop=True):
        
        # Each rank sends the nth chunk to the nth rank into its scratch space: chunks transpose operation
        # For Limited pair, Each rank sends nth chunk to nth "pre-determined" set of reducer ranks
         # Reducer ranks (could be 1 or more - equal to #GPU)
        for r1 in range(gpus):  
            for r2 in range(rsize):
                if r1 != gpuIds[r2]:
                    index = r2 * size
                    c = chunk(r1, Buffer.input, index, size) # Reference to the Source chunk
                    c.copy(gpuIds[r2], 'scratch', sendtb=gpuIds[r2], recvtb=r1)

        # Each reducer rank performs a local reduction on its nth chunk with all remote chunks in its scratch memory
        # Utilize 16 threadblocks for this reduction for better parallelism
        for r in range(rsize):
            for index in range(0, size * (gpus-1)): # Go through entire scratch memory, one chunk from one rank at a time and perfrom reduction
                c = chunk(gpuIds[r], Buffer.input, r*size + (index % size))
                c.reduce(chunk(gpuIds[r], 'scratch', index), sendtb=(index % gpus))
                                    
        
        # Reduce-Scatter phase done. All-Gather phase now
        # Each reducer rank sends the fully reduced nth chunk to all other gpus 
        for r1 in range(rsize): # Go through all reducer ranks (1 < r <= #GPUs) and send its reduced chunk
            for r2 in range(gpus):
                if gpuIds[r1] != r2:
                    index = r1 * size
                    c = chunk(gpuIds[r1], Buffer.input, index, size) # Reference to the Source chunk
                    c.copy(r2, Buffer.input, index, sendtb=r2, recvtb=gpuIds[r1])
                
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL128', 'LL'], help='Protocol')

args = parser.parse_args()

allreduce_ltd_pairs(args.num_gpus, args.instances, args.protocol)