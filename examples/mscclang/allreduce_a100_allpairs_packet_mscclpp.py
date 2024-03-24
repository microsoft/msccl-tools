# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def allreduce_allpairs(gpus, instances):
    size = gpus
    chunksperloop = gpus * gpus
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, True)
    with MSCCLProgram("allreduce_pairs", topology, collective, instances, protocol="LL",
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual, dependence_nop=True):

        # Each rank sends the nth chunk to the nth rank into scratch space
        for r1 in range(size):
            for r2 in range(size):
                if r1 != r2:
                    index = r2 * size
                    c = chunk(r1, Buffer.input, index, size=size)
                    c.put(r2, 'scratch', index=r1*size, sendtb=r2)

        # Each rank performs a local reduction on the nth chunk
        # Utilize 8 threadblocks for this reduction for better parallelism
        for r in range(size):
            for index in range(size):
                c = chunk(r, Buffer.input, r*size + index)
                for peer in range(size):
                    if peer != r:
                        c.reduce(chunk(r, 'scratch', peer*size+index), sendtb=index)

        # # Each rank sends the fully reduced nth chunk to all other gpus
        # for r1 in range(size):
        #     for r2 in range(size):
        #         index = r1 * size
        #         c = chunk(r1, Buffer.input, index + r2)
        #         for r3 in range(size):
        #             if r3 != r1:
        #                 c.put(r3, Buffer.input, index, sendtb=r2)

        Json()
        # Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')

args = parser.parse_args()

allreduce_allpairs(args.num_gpus, args.instances)
