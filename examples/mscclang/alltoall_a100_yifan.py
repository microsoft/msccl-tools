import argparse
import humanfriendly

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllToAll
from msccl.language.simulator import World, dgx2_top


def alltoall_hierarchical(num_nodes, gpus_per_node, size, protocol):
    World.set_top(dgx2_top)
    num_ranks = num_nodes * gpus_per_node
    topology = fully_connected(num_ranks)
    collective = AllToAll(num_ranks, 1, inplace=False)

        
    with MSCCLProgram("hierarchical_all_to_all", topology, collective, instances=-1, protocol=protocol):
        for n1 in range(num_nodes):
            for r in range(1,num_nodes):
                n2 = (n1 + r) % num_nodes
                # print(f"r {r} n1 {n1} n2 {n2}")

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

        # XML() # Simulate() # Prints the XML
        SearchBestSchedule(size)
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int, help ='number of nodes')
parser.add_argument('gpus_per_node', type=int, help ='gpus per node')
# parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('size', type=str, help='size at which to simulate')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()


alltoall_hierarchical(args.num_nodes, args.gpus_per_node, humanfriendly.parse_size(args.size), args.protocol)
