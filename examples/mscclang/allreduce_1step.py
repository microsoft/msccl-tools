# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def allreduce_allpairs(gpus, instances, protocol):
    size = gpus
    chunksperloop = gpus
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, True)
    with MSCCLProgram("allreduce_pairs", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual, dependence_nop=True):
        
        # Each rank sends the nth chunk to the nth rank into scratch space
        for rank in range(size):
            tb = 0
            for nghr in range(size):
                if rank != nghr:
                    c = chunk(rank, Buffer.input, index=0, size=size)
                    c.copy(nghr, 'scratch', sendtb=nghr, recvtb=rank)
                    tb += 1

        # Each rank performs a local reduction on the nth chunk
        # Utilize 8 threadblocks for this reduction for better parallelism
        for rank in range(size):
            index = 0
            tb = 0
            for nghr in range(size):
                if rank != nghr:
                    for s in range(size):
                        c = chunk(rank, Buffer.input, s)
                        c.reduce(chunk(rank, 'scratch', index), sendtb=s)
                        index += 1
                        tb += 1
                
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL128', 'LL'], help='Protocol')

args = parser.parse_args()

allreduce_allpairs(args.num_gpus, args.instances, args.protocol)