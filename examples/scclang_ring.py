# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.language import *
from sccl.topologies import *

size = 8
# A new program is created with a name and a topology
with SCCLProgram("allgather_ring", fully_connected(size)):
    # Loop over each chunk's root
    for r in range(size):
        # Get the rank, add an input with id r and mark it also as output
        c = Rank(r).input(r).output()
        next = r + 1 % size
        while next != r:
            # For each rank in the ring, send the chunk and mark it as output
            c = c.send(next).output()
            next = (next + 1) % size

with SCCLProgram("scheduling", line(2)):
    Rank(0).input(0).send(2).send(3).send(4).output()
    # Here wait(1) is used to avoid the two send(3)s happening at the same time
    Rank(1).input(1).send(2).wait(1).send(3).output()

size = 8
with SCCLProgram("allreduce_ring", ring(size)):
    for r in range(size):
        c = Rank(r).input(r)
        next = r + 1 % size
        while next != r:
            # There's something awkward about this reduce(c_next), it's
            # convenient to use the same id for this input as we did for c, but
            # we still need to say reduce(c_next). Would it be ok to not call
            # reduce here?
            c_next = Rank(next).input(r)
            c = c.send(next).reduce(c_next)
            next = (next + 1) % size
        c.output()
        while next != (r - 1) % size:
            # Chunks are only marked as output in this second loop here, where
            # the final values are produced.
            c = c.send(next).output()
            next = (next + 1) % size
