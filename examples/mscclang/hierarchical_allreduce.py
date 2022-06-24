# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

# Blue Connect style AllReduce https://proceedings.mlsys.org/paper/2019/file/9b8619251a19057cff70779273e95aa6-Paper.pdf
# Assumes only two-level switches

def ring_reduce_scatter(size, rank_offset=0, rank_step=1, local_chunk_size=1, chunk_offset=0, chunk_stride=1, chan=-1):
    for ch in range(0, size):
        index = ch * chunk_stride * local_chunk_size + chunk_offset
        for step in range(0, size-1):
            other = chunk(((step+1+ch) % size)*rank_step +rank_offset, Buffer.input, index, local_chunk_size)
            c = chunk(((step+2+ch) % size)*rank_step+rank_offset, Buffer.input, index, local_chunk_size)
            c.reduce(other, ch=chan)

def ring_all_gather(size, rank_offset=0, rank_step=1, local_chunk_size=1, chunk_offset=0, chunk_stride=1, chan=-1):
    for ch in range(0, size):
        index = ch * chunk_stride * local_chunk_size + chunk_offset
        for step in range(0, size-1):
            c = chunk(((step+ch) % size)*rank_step + rank_offset, Buffer.input, index, local_chunk_size)
            c.copy(((step+1+ch) % size)*rank_step + rank_offset, Buffer.input, index, ch=chan)

def hierarchical_allreduce(num_local_gpus, num_nodes, instances, protocol, schedule):
    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    collective = AllReduce(num_gpus, num_gpus, True)

    with MSCCLProgram("hierarchical_allreduce", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False):

        local_chunk_size = num_nodes
        if schedule == 'auto':
            for n in range(num_nodes):
                for offset in range(num_nodes):
                    ring_reduce_scatter(num_local_gpus, rank_offset=n * num_local_gpus, chunk_offset=offset, chunk_stride=num_nodes)

            # Cross node Reduce-Scatter
            for g in range(num_local_gpus):
                ring_reduce_scatter(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes)

            # Cross node All-gather
            for g in range(num_local_gpus):
                ring_all_gather(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes)


            # All gather within each node
            for n in range(num_nodes):
                for offset in range(num_nodes):
                    ring_all_gather(num_local_gpus, rank_offset=n * num_local_gpus, chunk_offset=offset, chunk_stride=num_nodes)

        else:
            # Reduce Scatter within each node
            for n in range(num_nodes):
                for offset in range(num_nodes):
                    ring_reduce_scatter(num_local_gpus, rank_offset=n * num_local_gpus, chunk_offset=offset, chunk_stride=num_nodes, chan=offset)

            # Cross node Reduce-Scatter
            for g in range(num_local_gpus):
                ring_reduce_scatter(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes, chan=g%2+num_nodes*2)

            # Cross node All-gather
            for g in range(num_local_gpus):
                ring_all_gather(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes, chan=g%2+num_nodes*2)


            # All gather within each node
            for n in range(num_nodes):
                for offset in range(num_nodes):
                    ring_all_gather(num_local_gpus, rank_offset=n * num_local_gpus, chunk_offset=offset, chunk_stride=num_nodes, chan=offset+num_nodes)

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help='number of gpus per node')
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL128', 'LL'], help='Protocol')
parser.add_argument('--schedule', type=str, default='auto', choices=['auto', 'manual'], help='Scheduling')

args = parser.parse_args()

hierarchical_allreduce(args.num_gpus, args.num_nodes, args.instances, args.protocol, args.schedule)

