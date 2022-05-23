# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllGather

def allgather_ring(size):
    topology = fully_connected(size)
    collective = AllGather(size, 1, False)
    with MSCCLProgram("allgather_ring", topology, collective, 1):
        # Loop over each chunk's root
        for r in range(size):
            # Get the chunk at rank r, input[r]
            c = chunk(r, Buffer.input, 0)
            # Copy chunk to the output buffer
            c = c.copy(r, buffer=Buffer.output, index=r, sendtb=0)

            next = (r + 1) % size
            while next != r:
                # For each rank in the ring, send the chunk to the next rank
                # Setting sender's tb and receiver's tb to be 0 so that send/receives on the
                # same rank can be merged into a receive-copy-send
                c = c.copy(next, buffer=Buffer.output, index=r)
                next = (next + 1) % size
        XML()
        Check()

def allgather_ring_inplace(size):
    topology = fully_connected(size)
    collective = AllGather(size, 1, True)
    with MSCCLProgram("allgather_ring", topology, collective, 1):
        # Loop over each chunk's root
        for r in range(size):
            # Get the chunk at rank r, input[r]
            c = chunk(r, Buffer.input, 0)

            next = (r + 1) % size
            while next != r:
                # For each rank in the ring, send the chunk to the next rank
                # Setting sender's tb and receiver's tb to be 0 so that send/receives on the
                # same rank can be merged into a receive-copy-send
                c = c.copy(next, buffer=Buffer.output, index=r)
                next = (next + 1) % size
        XML()
        Check()

allgather_ring(4)
# allgather_ring_inplace(4)