# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

def allreduce(instances):
    size = 8
    topology = fully_connected(size)
    collective = AllReduce(size, size * instances, False, "allreduce")
    with SCCLProgram("allreduce_ring", topology, collective, size * instances):
        for ch in range(instances):
            for r in range(size):
                # Get the chunk at rank r, input[r]
                index = r * instances + ch
                c = Rank(r).input(index)
                next = (r + 1) % size
                while next != r:
                    # For each rank in the ring, send the chunk to the next rank
                    c = c.reduce(next, buffer=Buffer.input, index=index, sendtb=r, recvtb=r, ch=ch)
                    next = (next + 1) % size
                while next != (r-1) % size:
                    c = c.send(next, buffer=Buffer.input, index=index, sendtb=r, recvtb=r, ch=ch)
                    next = (next + 1) % size
        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('instances', type=int, help='number of instances')
args = parser.parse_args()
allreduce(args.instances)