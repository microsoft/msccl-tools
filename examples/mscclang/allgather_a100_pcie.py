# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllGather

# Hierarchical allgather for A100 
def allgather_hier(gpus, instances, protocol):
    size = gpus
    chunksperloop = 1
    topology = fully_connected(gpus)
    collective = AllGather(size, chunksperloop, True)

    with MSCCLProgram("allgather_hierarchical", topology, collective, instances, protocol=protocol, 
        interleaved_replication=True, dependence_nop=True):
        for chnk in range(2):
            for r in range(size):
                if ((r % 2) == chnk):
                    c = chunk(r, Buffer.input, 0)
                    c.copy(r + 1 - 2 * chnk, Buffer.output, r)
            for r in range(size):
                if ((r % 2) == chnk):
                    c = chunk(r, Buffer.input, 0)
                    c.copy((r+2) % size, Buffer.output, r)
            for r in range(size):
                if ((r % 2) == chnk):
                    c = chunk(r, Buffer.output, (r+2) % size)
                    c.copy(r + 1 - 2 * chnk, Buffer.output, (r+2) % size)

        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL128', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

allgather_hier(args.num_gpus, args.instances, args.protocol)