# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def allreduce_allpairs(gpus, instances, protocol):
    size = gpus
    chunksperloop = 2
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, True)
    with MSCCLProgram("allreduce_ncv4", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False, dependence_nop=True):
        for chnk in range(chunksperloop):
            for r in range(size):
                if ((r % 2) == chnk):
                    c = chunk(r, Buffer.input, chnk)
                    c.reduce(chunk(r + 1 - 2 * chnk, Buffer.input, chnk))

            for r in range(size):
                if ((r % 2) == chnk):
                    c = chunk(r, Buffer.input, chnk)
                    c.copy((r+2) % size, 'scratch', chnk)
                    
            for r in range(size):
                if ((r % 2) == chnk):
                    c = chunk(r, Buffer.input, chnk)
                    c.reduce(chunk(r, 'scratch', chnk))
            
            for r in range(size):
                if ((r % 2) == chnk):
                    c = chunk(r, Buffer.input, chnk)
                    c.copy(r + 1 - 2 * chnk, Buffer.input, chnk)

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL128', 'LL'], help='Protocol')

args = parser.parse_args()

allreduce_allpairs(args.num_gpus, args.instances, args.protocol)