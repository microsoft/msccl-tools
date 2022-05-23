# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.language import *

# Ring all reduce for A100s
# Vary channels from [1-8] to divide parts of the ring over multiple channels/tbs.
# channels=1 is standard ring, all chunks are assigned to the same tb/channel
# channels=8 devotes 1 tb/channel to handling 1 chunk of the data
def allreduce_ring(size, channels):   
    # Reduce ring
    for step in range(0, size-1):
            for index in range(0, size):
                rank = (index + step) % size
                next_rank = (index + step + 1) % size
                channel = index%channels
                c = chunk(next_rank, Buffer.input, index)
                c.reduce(chunk(rank, Buffer.input, index), ch=channel, recvtb=channel, sendtb=channel)
    # Propagate ring
    for step in range(-1, size-2):
        for index in range(0, size):
            rank = (index + step) % size
            c = chunk(rank, Buffer.input, index)
            next_rank = (index + step + 1) % size
            channel = index%channels
            c = c.copy(next_rank, Buffer.input, index, ch=channel, recvtb=channel, sendtb=channel)