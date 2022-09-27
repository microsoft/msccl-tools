# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllGather

# https://web.cels.anl.gov/~thakur/papers/mpi-coll.pdf
def allgather_recursive_doubling(size, instances, protocol):
    topology = fully_connected(size)
    collective = AllGather(size, 1, True)
    with MSCCLProgram("allgather_recursive_doubling", topology, collective, instances, protocol=protocol, threadblock_policy=ThreadblockPolicy.manual):
        count = 1
        while count < size:
            # Every rank exchanges count chunks with neighbor count away
            for rank in range(size):
                peer = rank ^ count
                index = (rank // count) * count
                chunk(rank, Buffer.output, index, size=count).copy(peer, Buffer.output, index, sendtb=peer, recvtb=rank) 
            count *= 2

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help ='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

allgather_recursive_doubling(args.num_gpus, args.instances, args.protocol)
