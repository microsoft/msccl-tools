# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

def allreduce(instances):
    size = 8
    topology = fully_connected(size)
    collective = AllReduce(size, size * instances, True, "allreduce")
    with SCCLProgram("allreduce_ring", topology, collective, size * instances):
        
        for r in range(size):
            for i in range(instances):
                # Get the chunk at rank r, input[r]
                index = r * instances + i
                c = Rank(r).input(index)
                next = (r + 1) % size
                while next != r:
                    # For each rank in the ring, send the chunk to the next rank
                    c = c.reduce(next, Buffer.input, index, ch=r*instances+i, sendtb=r*instances+i, recvtb=r*instances+i)
                    next = (next + 1) % size
                while next != (r-1) % size:
                    c = c.send(next, Buffer.input, index, ch=r*instances+i, sendtb=r*instances+i, recvtb=r*instances+i)
                    next = (next + 1) % size
        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('instances', type=int, help='number of instances')
args = parser.parse_args()
allreduce(args.instances)