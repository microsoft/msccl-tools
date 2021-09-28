# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

# def allreduce_parallelrings(instances):
#     size = 8
#     topology = fully_connected(size)
#     collective = AllReduce(size, size * instances, True, "allreduce")
#     with SCCLProgram("allreduce_ring", topology, collective, size * instances, protocol="LL128"):
        
#         for r in range(size):
#             for i in range(instances):
#                 # Get the chunk at rank r, input[r]
#                 index = r * instances + i
#                 c = Rank(r).input(index)
#                 next = (r + 1) % size
#                 while next != r:
#                     # For each rank in the ring, send the chunk to the next rank
#                     c = c.reduce(next, Buffer.input, index, ch=r*instances+i, sendtb=r*instances+i, recvtb=r*instances+i)
#                     next = (next + 1) % size
#                 while next != (r-1) % size:
#                     c = c.send(next, Buffer.input, index, ch=r*instances+i, sendtb=r*instances+i, recvtb=r*instances+i)
#                     next = (next + 1) % size
#         XML()
#         Check()

# Ring all reduce for 
# Vary channels from [1-8] to divide parts of the ring over multiple channels.
# channels=1 is standard ring, all chunks are assigned to the same tb/channel
# channels=8 devotes 1 tb/channel to handling 1 chunk of the data
def allreduce_ring(instances, channels):
    size = 8
    topology = fully_connected(size)
    collective = AllReduce(size, size * instances, True, "allreduce")
    with SCCLProgram(f"allreduce_ring_{channels}channelsperring", topology, collective, size * instances, protocol="LL128"):
        
        
        for i in range(instances):
            # Reduce ring
            for step in range(0, size-1):
                for chunk in range(0, size):
                    index = chunk * instances + i
                    start = (chunk + step) % size
                    c = Rank(start).input(index)
                    next = (chunk + step + 1) % size
                    channel = (chunk%channels) * instances + i
                    c = c.reduce(next, Buffer.input, index, ch=channel, recvtb=channel, sendtb=channel)
            # Propagate ring
            for step in range(-1, size-2):
                for chunk in range(0, size):
                    index = chunk * instances + i
                    start = (chunk + step) % size
                    c = Rank(start).input(index)
                    next = (chunk + step + 1) % size
                    channel = (chunk%channels) * instances + i
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
parser.add_argument('version', type=int, help='which version to run, 0 = ring 1 = allpairs')
parser.add_argument('channels', type=int, help='Number of channels to use for 1 instance of the ring [1-8]')
parser.add_argument('instances', type=int, help='number of instances')

args = parser.parse_args()
if args.version == 0:
    allreduce_ring(args.instances, args.channels)
else:
    allreduce_pairs(args.instances)