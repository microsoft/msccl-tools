# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.language import *

def alltoall_hierarchical(num_nodes, gpus_per_node):
    num_ranks = num_nodes * gpus_per_node

    for n1 in range(num_nodes):
        for r in range(1,num_nodes):
            n2 = (n1 + r) % num_nodes

            # Gather all local chunks for the node neighbor
            for g1 in range(gpus_per_node):
                rank1 = n1 * gpus_per_node + g1

                for g2 in range(gpus_per_node):
                    rank2 = n1 * gpus_per_node + g2
                    # chunk to copy: g2 on n2
                    index = n2 * gpus_per_node + g2 
                    c = chunk(rank1, Buffer.input, index)
                    c = c.copy(rank2, f'copy_{n2}')

        for r in range(1,num_nodes):
            n2 = (n1 + r) % num_nodes
            # IB copy
            for g1 in range(gpus_per_node):
                rank = n1 * gpus_per_node + g1
                ib_peer = n2 * gpus_per_node + g1
                c = chunk(rank, f'copy_{n2}', 0, 8)
                c = c.copy(ib_peer, Buffer.output, c.get_dst_index(), ch=((n1+n2) % 8)*2+(rank%2)+2)

        
    # Handle local chunks within a node
    for rank in range(num_ranks):
        for g in range(gpus_per_node):
            index = (rank // gpus_per_node) * gpus_per_node + g
            c = chunk(rank, Buffer.input, index)
            c.copy(c.get_dst_rank(), Buffer.output, c.get_dst_index())
