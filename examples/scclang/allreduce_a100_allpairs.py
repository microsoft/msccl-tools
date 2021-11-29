# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

def allreduce_allpairs(instances, threadblocks):
    size = 8
    chunksperloop = 7
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, True, "allreduce")
    with SCCLProgram("allreduce_pairs", topology, collective, instances, protocol="LL", interleaved_replication=False, threadblocks=threadblocks):
        
        for r in range(size):
            for r1 in range(size):
                Rank(r).create_scratch(f'scratch{r1}') 

        instances = 1
        for i in range(instances):
            index = 7 * i
            for r1 in range(size):
                for r2 in range(size):
                    if r1 != r2:
                        c = Rank(r1).input(index, 7)
                        c = c.send(r2, f'scratch{r1}', index)

        for i in range(instances):
            for r1 in range(size):
                for r2 in range(size):
                    for chunk in range(0, 7):
                        if r1 != r2:
                            index = chunk * instances + i
                            c = Rank(r1).scratch(f'scratch{r2}', index)
                            c.reduce(r1, Buffer.input, index)
                
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('threadblocks', type=int, default=0, help='number of threadblocks per instance')

args = parser.parse_args()

allreduce_allpairs(args.instances, args.threadblocks)