# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from argparse import ArgumentParser
import humanfriendly
from msccl.language import * # type: ignore
from msccl.topologies import * # type: ignore
from msccl.language.collectives import AllGather # type: ignore
from msccl.language.simulator import dgx2_top, World # type: ignore

def allgather_ring(size):
    World.set_top(dgx2_top)
    topology = fully_connected(size)
    collective = AllGather(size, 1, False)
    with MSCCLProgram("allgather_ring", topology, collective, 12, instr_fusion=False):
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
        # Simulate(csv=False)
        Check()

def allgather_ring_search(num_gpus, size):
    World.set_top(dgx2_top)
    topology = fully_connected(num_gpus)
    collective = AllGather(num_gpus, 1, False)
    with MSCCLProgram("allgather_ring", topology, collective, 12, instr_fusion=False):
        # Loop over each chunk's root
        for r in range(num_gpus):
            # Get the chunk at rank r, input[r]
            c = chunk(r, Buffer.input, 0)
            # Copy chunk to the output buffer
            c = c.copy(r, buffer=Buffer.output, index=r, sendtb=0)

            next = (r + 1) % num_gpus
            while next != r:
                # For each rank in the ring, send the chunk to the next rank
                # Setting sender's tb and receiver's tb to be 0 so that send/receives on the
                # same rank can be merged into a receive-copy-send
                c = c.copy(next, buffer=Buffer.output, index=r)
                next = (next + 1) % num_gpus
        SearchBestSchedule(size)

def allgather_ring_inplace(size):
    topology = fully_connected(size)
    collective = AllGather(size, 1, True)
    with MSCCLProgram("allgather_ring", topology, collective, 12, instr_fusion=False):
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

parser = ArgumentParser()
parser.add_argument('num_gpus', type=int, help='number of gpus')
parser.add_argument('size', type=str, help='buffer size at which to simulate')
args = parser.parse_args()

allgather_ring_search(args.num_gpus, humanfriendly.parse_size(args.size))
# allgather_ring(4)
# allgather_ring_inplace(4)
