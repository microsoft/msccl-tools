# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

# https://web.cels.anl.gov/~thakur/papers/mpi-coll.pdf
def allreduce_distance_doubling(size, instances, protocol):
    topology = fully_connected(size)
    collective = AllReduce(size, 1, True)
    with SCCLProgram("allreduce_distance_doubling", topology, collective, instances, protocol=protocol):
        # TODO: We cannot express this algorithm - language sees a reduce as an atomic operation not (send, rrc)
        # where two reduces between the same slots on different ranks can happen simultenously because the
        # sends can be issued simultenously
        distance = 1
        while distance < size:
            for rank in range(size):
                peer = rank ^ distance
                chunk(rank, Buffer.output, 0).reduce(peer, Buffer.output, 0) 
            distance *= 2

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help ='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

allreduce_distance_doubling(args.num_gpus, args.instances, args.protocol)
