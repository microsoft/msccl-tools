import argparse

from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllToAll


def alltoall_hierarchical(num_nodes, gpus_per_node):
    num_ranks = num_nodes * gpus_per_node
    topology = fully_connected(num_ranks)
    collective = AllToAll(num_ranks, 1, inplace=False, name="alltoall")

        
    with SCCLProgram("hierarchical_all_to_all", topology, collective, 1):
        # Scratch space
        for rank in range(num_ranks):
            for node in range(num_nodes):
                Rank(rank).create_scratch(f'send_{node}')

        for n1 in range(num_nodes):
            for r in range(1,num_nodes):
                n2 = (n1 + r) % num_nodes
                # print(f"r {r} n1 {n1} n2 {n2}")

                # Gather all local chunks for the node neighbor
                for g1 in range(gpus_per_node):
                    rank1 = n1 * gpus_per_node + g1

                    for g2 in range(gpus_per_node):
                        rank2 = n1 * gpus_per_node + g2
                        # chunk to send: g2 on n2
                        index = n2 * gpus_per_node + g2 
                        chunk = Rank(rank1).input(index)
                        chunk = chunk.send(rank2, f'send_{n2}')
                        # print(f"Sending from {rank1} to {rank2}")
                        # chunk.print_chunk_info()

            for r in range(1,num_nodes):
                n2 = (n1 + r) % num_nodes
                # IB send
                for g1 in range(gpus_per_node):
                    rank = n1 * gpus_per_node + g1
                    ib_peer = n2 * gpus_per_node + g1
                    chunk = Rank(rank).scratch(f'send_{n2}', 0, 8)
                    chunk = chunk.send(ib_peer, Buffer.output, chunk.get_dst_index(), ch=(n2 % 8)*2+(rank%2)+2)

          
          # Handle local chunks within a node
        for rank in range(num_ranks):
            for g in range(gpus_per_node):
                index = (rank // gpus_per_node) * gpus_per_node + g
                chunk = Rank(rank).input(index)
                chunk.send(chunk.get_dst_rank(), Buffer.output, chunk.get_dst_index())


        XML() # Prints the XML
        Check()



parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int, help ='number of nodes')
parser.add_argument('gpus_per_node', type=int, help ='gpus per node')
args = parser.parse_args()


alltoall_hierarchical(args.num_nodes, args.gpus_per_node)
