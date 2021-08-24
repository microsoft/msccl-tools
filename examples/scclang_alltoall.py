# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from sccl.language import *
from sccl.topologies import *
from sccl.collectives import *

# Older version
def alltoall_hierarchical2(num_nodes, gpus_per_node, instances):
    num_ranks = num_nodes * gpus_per_node

    def NodeGpuPairFromRank(r):
        return (r//gpus_per_node, r%gpus_per_node)

    def RankFromNodeGpuPair(n, g):
        return n*gpus_per_node + g

    def CrossNodeNghr(node, g):
        nghrNode = g if node > g else g+1
        nghrG = node if nghrNode > node else node-1
        return nghrNode, nghrG

    def CrossNodeRouter(n1, n2):
        return (n2 if n1 > n2 else n2-1) % gpus_per_node

    def IBToUse(n1, n2):
        g1 = n2 if n1 > n2 else n2-1
        g2 = n1 if n2 > n1 else n1-1
        # invariant: CrossNodeNghr(n1, g1) == (n2, g2) and vice versa
        return g1, g2

    def AddChunk(ib_chunks, key, c):
        if key in ib_chunks: 
            ib_chunks[key] = ib_chunks[key].group(c)
        else:
            ib_chunks[key] = c
        
        
    # topology = NDv4(num_nodes, gpus_per_node)
    topology = fully_connected(num_ranks)
    s = 0 # Setting steps is hacky right now - actually specifies the relative ordering
    with SCCLProgram("hierarchical_all_to_all", topology, 'alltoall', instances):
        # Allocate scratch buffers for the local gathers - 2 scratch buffers for each node-node pair
        scratch_size = gpus_per_node * gpus_per_node * instances
        for n1 in range(num_nodes):
            for n2 in range(num_nodes):
                if n1 != n2:
                    h1 = CrossNodeRouter(n1, n2)
                    h2 = CrossNodeRouter(n2, n1)
                    r1 = RankFromNodeGpuPair(n1, h1)
                    r2 = RankFromNodeGpuPair(n2, h2)
                    Rank(r1).create_scratch((n1, n2), scratch_size) # Sender's buffer
                    Rank(r2).create_scratch((n1, n2), scratch_size) # Receiver's buffer
        ib_chunks = {}

        # Local Gathers
        # for ch in range(instances):
        for n1 in range(num_nodes):
            for g1 in range(gpus_per_node):
                for n2 in range(num_nodes):
                    for g2 in range(gpus_per_node):
                        r1 = RankFromNodeGpuPair(n1, g1)
                        r2 = RankFromNodeGpuPair(n2, g2)
                        c = Rank(r1).input(r2 * instances, instances)
                        
                        # All chunks destined to n2 should be sent together
                        # in case of h1 = g1, do a copy to scratch buffer
                        if (n1 != n2): # general case - gather in scratch to route through IB
                            h1 = CrossNodeRouter(n1, n2)
                            h2 = CrossNodeRouter(n2, n1)
                            next = RankFromNodeGpuPair(n1, h1)
                            scratch_key = (n1, n2)
                            scratch_index = (g1 * gpus_per_node + g2) * instances
                            c = c.send(next, step=s, buffer=scratch_key, index=scratch_index, ch=0)
                            AddChunk(ib_chunks, scratch_key, c)
                        elif (g1 != g2):
                            c.send(r2, buffer=Buffer.output, index=r1*instances, step=s, ch=0) # this should be coalesced with the first send above
                        else:
                            c.send(r1, step=s, buffer=Buffer.output, index=r1*instances, ch=0) # copy input to output.
                        s += 1


        # # IB Send. All chunks from all local nodes destined to n2 should be sent together
        for key, ib_chunk in ib_chunks.items(): 
            (n1, n2) = key
            h2 = CrossNodeRouter(n2, n1)
            next2 = RankFromNodeGpuPair(n2, h2)
            ib_chunks[key] = ib_chunk.send(next2, step=s, buffer=key, ch=0)
            s +=1

        # # Local scatter within the nodes
        for key, ib_chunk in ib_chunks.items(): 
            n1, n2 = key
            chunks = ib_chunk.split(gpus_per_node)
            for g1, cc in enumerate(chunks):
                c = cc.split(gpus_per_node)
                for g2, cc in enumerate(c):
                    next3 = RankFromNodeGpuPair(n2, g2)
                    index = (n1 * gpus_per_node + g1) * instances
                    cc.send(next3, step=s, buffer=Buffer.output, index=index, ch=1)
                    s +=1

        XML() # Prints the XML
        Check()

def alltoall_hierarchical(num_nodes, gpus_per_node, instances):
    num_ranks = num_nodes * gpus_per_node

    # (node, local rank) to rank
    def RankFromNodeGpuPair(n, g):
        return n*gpus_per_node + g

    # (node, local rank) to IB buddy (node, local rank)
    def CrossNodeNghr(node, g):
        nghrNode = g if node > g else g+1
        nghrG = node if nghrNode > node else node-1
        return nghrNode, nghrG

    # For cross node traffic from node n1 to node n2, returns the local rank
    # of the IB pair on node n1
    def CrossNodeRouter(n1, n2):
        return (n2 if n1 > n2 else n2-1) % gpus_per_node

    # Groups chunk reference into one large chunk reference (used for IB)
    # Save them under a key in the dictionary ib_chunks
    def AddChunk(ib_chunks, key, c):
        if key in ib_chunks: 
            ib_chunks[key] = ib_chunks[key].group(c)
        else:
            ib_chunks[key] = c
        

    topology = fully_connected(num_ranks)
    s = 0 # Setting steps is hacky right now - used to specify the relative ordering within a tb
    
    with SCCLProgram("hierarchical_all_to_all", topology, 'alltoall', instances):
        # Allocate scratch buffers to gather chunks to be sent over IB
        # 2 scratch buffers for each node-node pair for the sender and receiver
        for n1 in range(num_nodes):
            for n2 in range(num_nodes):
                if n1 != n2:
                    h1 = CrossNodeRouter(n1, n2)
                    h2 = CrossNodeRouter(n2, n1)
                    r1 = RankFromNodeGpuPair(n1, h1)
                    r2 = RankFromNodeGpuPair(n2, h2)
                    Rank(r1).create_scratch((n1, n2)) # Sender's buffer named (n1, n2)
                    Rank(r2).create_scratch((n1, n2)) # Receiver's buffer named (n1, n2)
        ib_chunks = {} # Keeps track of chunks going over IB buffer buffer name -> chunk

        # Local Gathers
        for n1 in range(num_nodes):
            for g1 in range(gpus_per_node):
                for n2 in range(num_nodes):
                    for g2 in range(gpus_per_node):
                        for ch in range(instances):
                            r1 = RankFromNodeGpuPair(n1, g1)
                            r2 = RankFromNodeGpuPair(n2, g2)
                            # Rank(r) gives accesses the rth rank of the program
                            # .input(index) gives a reference to the chunk at that index
                            c = Rank(r1).input(r2 * instances + ch)
                            
                            if (n1 != n2): 
                                # Gather chunks destined for cross node ranks in scratch to route through IB
                                h1 = CrossNodeRouter(n1, n2)
                                h2 = CrossNodeRouter(n2, n1)
                                gather_rank = RankFromNodeGpuPair(n1, h1)
                                buffer_key = (n1, n2)
                                buffer_index = (g1 * gpus_per_node + g2) * instances + ch
                                # Send chunk to the gather_rank. Send returns a chunk reference to the 
                                # receiver's chunk
                                c = c.send(gather_rank, step=s, buffer=buffer_key, index=buffer_index, ch=0)
                                AddChunk(ib_chunks, buffer_key, c) # Group the chunks using a particular IB pair into one large chunk reference
                            elif (g1 != g2):
                                # Directly send chunks destined for ranks within the node
                                c.send(r2, buffer=Buffer.output, index=r1*instances+ch, step=s, ch=0) 
                            else:
                                # Copy chunks destined for current rank into the output buffer
                                c.send(r1, step=s, buffer=Buffer.output, index=r1*instances+ch, ch=0)
                            s += 1


        # IB Send and local scatters
        for buffer_key, ib_chunk in ib_chunks.items(): 
            (n1, n2) = buffer_key
            h2 = CrossNodeRouter(n2, n1)
            ib_rank = RankFromNodeGpuPair(n2, h2)
            # IB send divided across multiple parallel channels
            chunks = ib_chunk.split(instances)
            for ch, chunk in enumerate(chunks):
                chunk = chunk.send(ib_rank, step=s, buffer=buffer_key, ch=ch)
                # Local scatter
                cs = chunk.split(gpus_per_node * gpus_per_node)
                for i, c in enumerate(cs):
                    # Access the chunk's origin rank and index to determine its final rank and index.
                    origin_index = c.get_origin_index()
                    origin_rank = c.get_origin_rank()
                    final_rank = origin_index // instances
                    index = origin_rank * instances + origin_index % instances
                    c.send(final_rank, step=s, buffer=Buffer.output, index=index, ch= 1)
                    s +=1

        XML() # Prints the XML
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int, help ='number of nodes')
parser.add_argument('gpus_per_node', type=int, help ='gpus per node')
parser.add_argument('instances', type=int)
args = parser.parse_args()


alltoall_hierarchical(args.num_nodes, args.gpus_per_node, args.instances)