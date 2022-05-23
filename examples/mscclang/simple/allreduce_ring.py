# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllReduce


def allreduce_ring(size, instances):
    # Logical topology
    topology = fully_connected(size)
    collective = AllReduce(size, size, inplace=True)

    with MSCCLProgram("allreduce_ring_inplace", topology, collective, instances):
        for r in range(size):
            index = r
            # (rank, buffer, index)
            c = chunk(r, Buffer.input, index)
            next = (r + 1) % size
            # Chunk travels around the ring being reduced
            while next != r:
                c1 = chunk(next, buffer=Buffer.input, index=r)
                # c1 += c
                c = c1.reduce(c)
                next = (next + 1) % size
            
            # Send the fully reduced chunk around the ring
            while next != (r - 1) % size:
                c = c.copy(next, buffer=Buffer.input, index=r)
                next = (next + 1) % size

        Check()
        XML()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')

args = parser.parse_args()

allreduce_ring(args.num_gpus, args.instances)
