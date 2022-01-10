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
def allreduce_ring(size, channels):   
    # Reduce ring
    for step in range(0, size-1):
        for index in range(0, size):
            rank = (index + step) % size
            c = chunk(Buffer.input, rank, index)
            next_rank = (index + step + 1) % size
            channel = index%channels
            c = c.reduce(next_rank, Buffer.input, index, ch=channel, recvtb=channel, sendtb=channel)
    # Propagate ring
    for step in range(-1, size-2):
        for index in range(0, size):
            rank = (index + step) % size
            c = chunk(Buffer.input, rank, index)
            next_rank = (index + step + 1) % size
            channel = index%channels
            c = c.send(next_rank, Buffer.input, index, ch=channel, recvtb=channel, sendtb=channel)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('channels', type=int, help='Number of channels to use for 1 instance of the ring [1-8]')
    parser.add_argument('instances', type=int, help='number of instances')
    args = parser.parse_args()

    num_ranks = 8
    topology = fully_connected(num_ranks)
    collective = AllReduce(num_ranks, num_ranks, True, "allreduce")
    with SCCLProgram(f"allreduce_ring_{args.channels}channelsperring", topology, collective, args.instances,
        protocol="LL128", threadblock_policy=ThreadblockPolicy.manual):
        allreduce_ring(num_ranks, args.channels)
        Check()
        XML()
