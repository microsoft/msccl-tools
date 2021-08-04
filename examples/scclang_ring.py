# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.language import *
from sccl.topologies import *
from sccl.collectives import *

def allgather_ring(size):
    # A new program is created with a name and a topology, desired collective
    collective = allgather(size)
    topology = fully_connected(size)
    with SCCLProgram("allgather_ring", collective, topology):
        # Loop over each chunk's root
        for r in range(size):
            # Get the chunk at rank r, input[r]
            c = Rank(r).input(r)
            # Copy chunk to the output buffer
            c.copyto(step=0, tb=0)

            next = (r + 1) % size
            hops = 0
            while next != r:
                # For each rank in the ring, send the chunk to the next rank
                c = c.send(next, step=hops, sendtb=1, recvtb=2)
                next = (next + 1) % size
                hops += 1
        # Print()
        XML()
        assert Check() # check desired chunks end up on each rank

def wait():
    with SCCLProgram("scheduling", line(2)):
        Rank(0).input(0).send(2).send(3).send(4).output()
        # Here wait(1) is used to avoid the two send(3)s happening at the same time
        Rank(1).input(1).send(2).wait(1).send(3).output()

def allreduce_ring():
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


    
def alltoall_hierarchical(num_nodes, gpus_per_node):
    num_ranks = num_nodes * gpus_per_node
    def NodeGpuPairFromRank(r):
        return (r//gpus_per_node, r%gpus_per_node)

    def RankFromNodeGpuPair(n, g):
        return n*gpus_per_node + g

    def CrossNodeNghr(node, g):
        nghrNode = g if node > g else g+1
        nghrG = node if nghrNode > node else node-1
        return nghrNode, nghrG

    def IBToUse(n1, n2):
        g1 = n2 if n1 > n2 else n2-1
        g2 = n1 if n2 > n1 else n1-1
        # invariant: CrossNodeNghr(n1, g1) == (n2, g2) and vice versa
        return g1, g2
        
    # topology = NDv4(num_nodes, gpus_per_node)
    topology = fully_connected(num_ranks)
    collective = alltoall(num_ranks)
    s = 0 # Setting steps is hacky right now
    with SCCLProgram("hierarchical_all_to_all", collective, topology):
        for r1 in range(num_ranks):
            for r2 in range(num_ranks):
                chunk = r2 * num_ranks + r1
                c = Rank(r1).input(chunk)
                (n1, g1) = NodeGpuPairFromRank(r1)
                (n2, g2) = NodeGpuPairFromRank(r2)
                if (n1 != n2): # general case - route through IB
                    (h1, h2) = IBToUse(n1, n2)  # use the IB link from (n1, h1) to (n2, h2) for this chunk
                    
                    # Local Gather. 
                    # All chunks destined to n2 should be sent together
                    # in case of h1 = g1, do a copy to scratch buffer
                    next = RankFromNodeGpuPair(n1, h1)
                    # print(f'Chunk {chunk} goes from {r1} to {next} using IBs {h1} {h2}')
                    if next != r1:
                        c = c.send(next, step=s)
                        s+=1

                    # IB Send. All chunks from all local nodes destined to n2 should be sent together   
                    next2 = RankFromNodeGpuPair(n2, h2)
                    c = c.send(next2, step=s)
                    s +=1

                    # Local Scatter
                    next3 = RankFromNodeGpuPair(n2, g2)
                    if next2 != next3:
                        c = c.send(next3, step=s)
                        s +=1
                elif (g1 != g2):
                    c = c.send(r2, step=s) # this should be coalesced with the first send above
                    s +=1
                else:
                    c.copyto(tb=0, step=0)
                    # copy input to output.
        XML()
        Check()
allgather_ring(8)
alltoall_hierarchical(4, 3)