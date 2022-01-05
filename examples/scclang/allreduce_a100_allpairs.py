# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

def allreduce_allpairs(instances):
    size = 8
    chunksperloop = 8
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, True, "allreduce")
    with SCCLProgram("allreduce_pairs", topology, collective, instances, protocol="LL", 
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual):
        
        for r in range(size):
            create_scratch(r, 'scratch') 

        # Each rank sends the nth chunk to the nth rank into scratch space
        for r1 in range(size):
            for r2 in range(size):
                if r1 != r2:
                    index = r2
                    c = chunk(Buffer.input, r1, index)
                    c.send(r2, 'scratch', sendtb=r2, recvtb=r1, ch=0)

        # Each rank performs a local reduction on the nth chunk
        for r in range(size):
            for index in range(0, 7):
                c = chunk('scratch', r, index)
                c.reduce(r, Buffer.input, r, sendtb=r, ch=0)
        
        # Each rank sends the fully reduced nth chunk to all other gpus
        for r1 in range(size):
            for r2 in range(size):
                if r1 != r2:
                    index = r1
                    c = chunk(Buffer.input, r1, index)
                    c.send(r2, Buffer.input, index, sendtb=r2, recvtb=r1, ch=0)
                
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('instances', type=int, help='number of instances')
# parser.add_argument('threadblocks', type=int, default=0, help='number of threadblocks per instance')

args = parser.parse_args()

allreduce_allpairs(args.instances)