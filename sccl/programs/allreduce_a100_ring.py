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
            c1 = chunk(rank, Buffer.input, index)
            next_rank = (index + step + 1) % size
            channel = index%channels
            c = chunk(next_rank, Buffer.input, index)
            c.reduce(c1, ch=channel, recvtb=channel, sendtb=channel)
    # Propagate ring
    for step in range(-1, size-2):
        for index in range(0, size):
            rank = (index + step) % size
            c = chunk(rank, Buffer.input, index)
            next_rank = (index + step + 1) % size
            channel = index%channels
            c = c.copy(next_rank, Buffer.input, index, ch=channel, recvtb=channel, sendtb=channel)