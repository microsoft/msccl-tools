# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllGather

# https://web.cels.anl.gov/~thakur/papers/mpi-coll.pdf
def allgather_recursive_doubling(size, instances, protocol):
    topology = fully_connected(size)
    collective = AllGather(size, instances, True)
    with SCCLProgram("allgather_recursive_doubling", topology, collective, 1, protocol=protocol):
        count = 1
        while count < size:
            # Every rank exchanges count chunks with neighbor count away
            for rank in range(size):
                for i in range(instances):
                    peer = rank ^ count
                    index = ((rank // count) * count) * instances + i * count
                    chunk(rank, Buffer.output, index, size=count).send(peer, Buffer.output, index) 
            count *= 2

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help ='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

allgather_recursive_doubling(args.num_gpus, args.instances, args.protocol)
