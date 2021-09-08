# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from sccl.language import *
from sccl.topologies import *
from sccl.collectives import *

def allreduce_ring(size, instances):
    topology = fully_connected(size)
    with SCCLProgram("allreduce_ring_inplace", topology, 'allreduce', instances):
        s = 0
        for r in range(size):
            for ch in range(instances):
                index = ch + r*instances
                c = Rank(r).input(index)
                next = (r + 1) % size
                # Chunk travels around the ring being reduced
                while next != r:
                    # TODO: Do we want reduce to be chunka.reduce(chunkb) -> chunkab on b's rank (argument is other chunk)
                    # or chunka.reduce(dst-rank, dst-index) -> chunkab on dst-rank  (argument is location of other chunk)
                    c = c.reduce(next, buffer=Buffer.input, index=ch + r*instances, step=s, ch=ch)
                    next = (next + 1) % size
                    s += 1
                
                # Send the fully reduced chunk around the ring
                while next != (r - 1) % size:
                    c = c.send(next, buffer=Buffer.input, index=ch + r*instances, step=s, ch=ch)
                    next = (next + 1) % size
                    s += 1

        Check()
        XML()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
args = parser.parse_args()

allreduce_ring(args.num_gpus, args.instances)
