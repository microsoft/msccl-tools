# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllGather

def const_func(x):
    def f(index): return x
    return f

def alternate(x, offset=0):
    def f(index): return (index % x) + offset
    return f

def ring_reduce_scatter(size, rank_offset=0, rank_step=1, local_chunk_size=1, chunk_offset=0, chunk_stride=1, chan=const_func(-1)):
    for ch in range(0, size):
        index = ch * chunk_stride * local_chunk_size + chunk_offset
        for step in range(0, size-1):
            other = chunk(((step+1+ch) % size)*rank_step +rank_offset, Buffer.output, index, local_chunk_size)
            c = chunk(((step+2+ch) % size)*rank_step+rank_offset, Buffer.output, index, local_chunk_size)
            c.reduce(other, ch=chan(index))

def ring_all_gather(size, rank_offset=0, rank_step=1, local_chunk_size=1, chunk_offset=0, chunk_stride=1, chan=const_func(-1)):
    for ch in range(0, size):
        index = ch * chunk_stride * local_chunk_size + chunk_offset
        for step in range(0, size-1):
            c = chunk(((step+ch) % size)*rank_step + rank_offset, Buffer.output, index, local_chunk_size)
            c.copy(((step+1+ch) % size)*rank_step + rank_offset, Buffer.output, index, ch=chan(index))

def hierarchical_allgather(num_local_gpus, num_nodes, instances, protocol, intra_ch):
    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    collective = AllGather(num_gpus, 1, True)

    with MSCCLProgram("hierarchical_allgather", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False):

        local_chunk_size = num_nodes
        # Cross node All-gather
        for g in range(num_local_gpus):
            ring_all_gather(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g, chunk_stride=num_local_gpus, chan=const_func(g%2+num_nodes*intra_ch))


        # All gather within each node
        for n in range(num_nodes):
            for offset in range(num_nodes):
                ring_all_gather(num_local_gpus, rank_offset=n * num_local_gpus, chunk_offset=offset*num_local_gpus, chan=alternate(intra_ch, offset*intra_ch))

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help='number of gpus per node')
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('channels', type=int, help='number of channels per intra_node ring')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL128', 'LL'], help='Protocol')

args = parser.parse_args()

hierarchical_allgather(args.num_gpus, args.num_nodes, args.instances, args.protocol, args.channels)

