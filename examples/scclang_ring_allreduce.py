# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

def allreduce_parallelrings(instances):
    size = 8
    topology = fully_connected(size)
    collective = AllReduce(size, size * instances, True, "allreduce")
    with SCCLProgram("allreduce_ring", topology, collective, size * instances, protocol="LL128"):
        
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

def allreduce_standard(instances):
    size = 8
    topology = fully_connected(size)
    collective = AllReduce(size, size * instances, True, "allreduce")
    with SCCLProgram("allreduce_ring", topology, collective, size * instances, protocol="LL128"):
        
        
        for i in range(instances):
            # Get the chunk at rank r, input[r]
            for step in range(0, size-1):
                for chunk in range(0, size):
                    index = chunk * instances + i
                    start = (chunk + step) % size
                    c = Rank(start).input(index)
                    next = (chunk + step + 1) % size
                    channel = (chunk%2) * instances + i
                    c = c.reduce(next, Buffer.input, index, ch=channel, recvtb=channel, sendtb=channel)

            for step in range(-1, size-2):
                for chunk in range(0, size):
                    index = chunk * instances + i
                    start = (chunk + step) % size
                    c = Rank(start).input(index)
                    next = (chunk + step + 1) % size
                    channel = (chunk%2) * instances + i
                    c = c.send(next, Buffer.input, index, ch=channel, recvtb=channel, sendtb=channel)
               
               
        XML()
        Check()

def allreduce_pairs(instances):
    size = 8
    topology = fully_connected(size)
    collective = AllReduce(size, 7 * instances, True, "allreduce")
    with SCCLProgram("allreduce_pairs", topology, collective, size * instances, protocol="LL128"):
        
        for r in range(size):
            for r1 in range(size):
                Rank(r).create_scratch(f'scratch{r1}') 

        for i in range(instances):
            index = 7 * i
            for r1 in range(size):
                for r2 in range(size):
                    if r1 != r2:
                        c = Rank(r1).input(index, 7)
                        c = c.send(r2, f'scratch{r1}', index, ch=i, sendtb=r2*instances + i, recvtb=r1 * instances + i)

        for i in range(instances):
            for r1 in range(size):
                for r2 in range(size):
                    for chunk in range(0, 7):
                        if r1 != r2:
                            index = chunk * instances + i
                            c = Rank(r1).scratch(f'scratch{r2}', index)
                            c.reduce(r1, Buffer.input, index, ch=i, sendtb=r2 * instances + i)
                
        XML()
        Check()



parser = argparse.ArgumentParser()
parser.add_argument('version', type=int, help='which ring to run')
parser.add_argument('instances', type=int, help='number of instances')

args = parser.parse_args()
if args.version == 0:
    allreduce_standard(args.instances)
elif args.version == 1:
    allreduce_parallelrings(args.instances)
else:
    allreduce_pairs(args.instances)