# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

# Ring all reduce for A100s
# Vary channels from [1-8] to divide parts of the ring over multiple channels/tbs.
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


parser = argparse.ArgumentParser()
parser.add_argument('channels', type=int, help='Number of channels to use for 1 instance of the ring [1-8]')
parser.add_argument('instances', type=int, help='number of instances')
args = parser.parse_args()

allreduce_ring(args.instances, args.channels)
