# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import *

def allgather_ring_inplace(gpus, gpu_offset=0, index_offset=0, ch=0):
    for rank in range(gpu_offset, gpu_offset+gpus):
            index = index_offset + rank - gpu_offset
            c = chunk(rank, Buffer.input, 0)
            for r_next in range(1, gpus):
                next_rank = (rank + r_next) % gpus + gpu_offset
                c = c.copy(next_rank, Buffer.output, index, ch=0)

def allreduce_ring_inplace(gpus, gpu_offset=0, index_offset=0, ch=0):
    for rank in range(gpu_offset, gpu_offset+gpus):
        index = index_offset + rank - gpu_offset
        c = chunk(rank, Buffer.input, index)
        # Reduce ring
        for r_next in range(1, gpus):
            next_rank = (rank + r_next) % gpus + gpu_offset
            c = chunk(next_rank, Buffer.input, index).reduce(c, ch=ch)
        # Propagate ring
        for r_next in range(0, gpus-1):
            next_rank = (rank + r_next) % gpus + gpu_offset
            c = c.copy(next_rank, Buffer.input, index, ch=ch)
    