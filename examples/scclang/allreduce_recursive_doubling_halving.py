# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import argparse

from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

def reduce_scatter_vector_halving_distance_doubling(size):
    count = size // 2
    while count >= 1:
        for rank in range(size):
            peer = rank ^ count
            index = ((peer // count) * count)
            chunk(rank, Buffer.input, index, size=count).reduce(peer, Buffer.output, index, sendtb=peer, recvtb=rank, ch=0)
        count //= 2

def allgather_recursive_vector_doubling_distance_halving(size):
    count = 1
    while count < size:
        for rank in range(size):
            peer = rank ^ count
            index = ((rank // count) * count)
            chunk(rank, Buffer.output, index, size=count).send(peer, Buffer.output, index, sendtb=peer, recvtb=rank, ch=0) 
        count *= 2

def allreduce(size, instances, protocol):
    topology = fully_connected(size)
    logical_chunk = size
    collective = AllReduce(size, logical_chunk, True)
    with SCCLProgram("allreduce_recursive_doubling_halving", topology, collective, instances, protocol,
         interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual):
        reduce_scatter_vector_halving_distance_doubling(size)
        allgather_recursive_vector_doubling_distance_halving(size)
        XML()
        Check()
        Print()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allreduce(args.num_gpus, args.instances, args.protocol)
