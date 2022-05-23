# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.language import *

def alltoall_three_step(num_nodes, gpus_per_node, instances=1, ib_connections=1):
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
    
    ib_chunks = {} # Keeps track of chunks going over IB buffer buffer name -> chunk
    for n1 in range(num_nodes):
        for g1 in range(gpus_per_node):
            for ch in range(instances):
                for n2 in range(num_nodes):
                    r1 = RankFromNodeGpuPair(n1, g1)
                    if (n1 != n2): 
                        # Send over all chunks destined for that node to the peer gpu that handles chunks to that node
                        c = chunk(r1, Buffer.input, n2 * gpus_per_node * instances + ch * gpus_per_node, gpus_per_node)
                        # Gather chunks destined for cross node ranks in scratch to route through IB
                        gather_rank, _ = CrossNodeGpus(n1, n2)
                        buffer_key = (n1, n2)
                        # Send chunk to the gather_rank. Send returns a chunk reference to the 
                        # receiver's chunk
                        c = c.copy(gather_rank, buffer=buffer_key, ch=ch*2)
                        # Group the chunks using a particular IB pair into one large chunk reference
                        AddChunk(ib_chunks, buffer_key, c) 
                    else:
                        # Within a node - direct copy/copy the chunks over nvlink to the output buffer. 
                        # Use a different channel to ensure that we don't get in the way of copys/receives above
                        # which are on the critical path.
                        for g2 in range(gpus_per_node):
                            r2 = RankFromNodeGpuPair(n2, g2)
                            c = chunk(r1, Buffer.input, r2 * instances + ch)
                            c.copy(r2, buffer=Buffer.output, index=c.get_dst_index(), ch=ch*2)

                

    # IB Send and local scatters
    for buffer_key, ib_chunk in ib_chunks.items(): 
        (n1, n2) = buffer_key
        _, scatter_rank = CrossNodeGpus(n1, n2)
        # IB copy divided across multiple parallel channels
        chunks = ib_chunk.split(ib_connections)
        for ch, c in enumerate(chunks):
            # Note: If we are only going to use 1 IB connection for each IB copy
            # alternate between channels 0 and 1 to utilize both IB links.
            if ib_connections == 1:
                ib_channel = c.rank % 2
            else:
                ib_channel = ch
            c = c.copy(scatter_rank, buffer=buffer_key, ch=ib_channel)
            # Local scatter
            cs = c.split(gpus_per_node * gpus_per_node)
            for i, c in enumerate(cs):
                # Access the chunk's destination rank and index to route it to its final place
                final_rank = c.get_dst_rank()
                index = c.get_dst_index()
                c.copy(final_rank, buffer=Buffer.output, index=index, ch=ch*2 + 1)