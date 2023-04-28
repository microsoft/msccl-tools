# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllGather

# Allpairs allgather for A100 
def allgather_allpairs(gpus, instances, protocol):
    size = gpus
    topology = fully_connected(gpus)
    collective = AllGather(size, size, True)

    with MSCCLProgram(f"allgather_allpairs", topology, collective, instances,
         protocol=protocol, threadblock_policy=ThreadblockPolicy.manual):
        
        # Each rank sends its nth chunk to all other gpus
        for r1 in range(gpus):
            for r2 in range(gpus):
                if r1 != r2:
                    index = 0
                    c = chunk(r1, Buffer.input, index, size)
                    c.copy(r2, Buffer.input, index, sendtb=r2, recvtb=r1)
        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL128', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

allgather_allpairs(args.num_gpus, args.instances, args.protocol)