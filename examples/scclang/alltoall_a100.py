# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllToAll

def alltoall_hierarchical(num_nodes, gpus_per_node, instances, ib_channels):
    num_ranks = num_nodes * gpus_per_node

    # (node, local gpu) to rank
    # (n, g) => r
    def RankFromNodeGpuPair(n, g):
        return n*gpus_per_node + g

    # For cross node traffic from node n1 to node n2, returns the ranks g
    # gpus on n1 and n2 that handle that traffic.
    def CrossNodeGpus(n1, n2):
        def LocalRank(n1, n2):
            return (n2 if n1 > n2 else n2-1) % gpus_per_node
        r1 = RankFromNodeGpuPair(n1, LocalRank(n1, n2))
        r2 = RankFromNodeGpuPair(n2, LocalRank(n2, n1))
        return (r1, r2)

    # Groups chunk reference into one large chunk reference (used for IB)
    # Save them under a key in the dictionary ib_chunks
    def AddChunk(ib_chunks, key, c):
        if key in ib_chunks: 
            ib_chunks[key] = ib_chunks[key].group(c)
        else:
            ib_chunks[key] = c
        

    topology = fully_connected(num_ranks)
    collective = AllToAll(num_ranks, instances, inplace=False, name="alltoall")
    
    with SCCLProgram("hierarchical_all_to_all", topology, collective, instances):
        # Allocate scratch buffers to gather chunks to be sent over IB
        # 2 scratch buffers for each node-node pair for the sender and receiver
        for n1 in range(num_nodes):
            for n2 in range(num_nodes):
                if n1 != n2:
                    # Determine which are the two ranks that handle this cross node traffic
                    r1, r2 = CrossNodeGpus(n1, n2)
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
                            # input(i) gives a reference to ith chunk
                            c = Rank(r1).input(r2 * instances + ch)
                            
                            if (n1 != n2): 
                                # Gather chunks destined for cross node ranks in scratch to route through IB
                                gather_rank, _ = CrossNodeGpus(n1, n2)
                                buffer_key = (n1, n2)
                                # Send chunk to the gather_rank. Send returns a chunk reference to the 
                                # receiver's chunk
                                c = c.send(gather_rank, buffer=buffer_key, ch=ch)
                                # Group the chunks using a particular IB pair into one large chunk reference
                                AddChunk(ib_chunks, buffer_key, c) 
                            else:
                                # Directly send chunks destined for ranks within the node or
                                # copy chunks destined for current rank into the output buffer
                                c.send(r2, buffer=Buffer.output, index=c.get_dst_index(), ch=ch)

        # IB Send and local scatters
        for buffer_key, ib_chunk in ib_chunks.items(): 
            (n1, n2) = buffer_key
            _, scatter_rank = CrossNodeGpus(n1, n2)
            # IB send divided across multiple parallel channels
            chunks = ib_chunk.split(ib_channels)
            for i, chunk in enumerate(chunks):
                # IB send
                # Make certain even number ranks send on even channels and odd number ranks send on odd channels
                # to utilize both IB cards 
                ib_channel = i*2 + (chunk.rank % 2)
                chunk = chunk.send(scatter_rank, buffer=buffer_key, ch=ib_channel)

                # Local scatter
                cs = chunk.split(gpus_per_node * gpus_per_node)
                for i, c in enumerate(cs):
                    # Access the chunk's destination rank and index to route it to its final place
                    final_rank = c.get_dst_rank()
                    index = c.get_dst_index()
                    c.send(final_rank, buffer=Buffer.output, index=index, ch=ch)

        XML() # Prints the XML
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int, help ='number of nodes')
parser.add_argument('gpus_per_node', type=int, help ='gpus per node')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--ib_channels', type=int, default=-1, help='Number of connections used for each IB send. Default: number of instances')
args = parser.parse_args()

if args.ib_channels == -1:
    args.ib_channels = args.instances

alltoall_hierarchical(args.num_nodes, args.gpus_per_node, args.instances, args.ib_channels)