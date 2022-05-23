# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

# Binomial tree and mirrored binomial tree
# Mirrored trees adopted from: http://algo2.iti.kit.edu/documents/2tree.pdf
def allreduce_binomial_tree(size, instances, trees, protocol):
    topology = fully_connected(size)
    collective = AllReduce(size, trees, True)
    with MSCCLProgram("allreduce_binomial_tree", topology, collective, instances, protocol=protocol):
        distance = 1
        # Reduce tree - reducing onto Rank 0
        while distance <= size // 2:
            # Reduce onto the left neighbor that is distance away
            for rank in range(0, size, distance*2):
                peer = rank + distance
                c1 = chunk(peer, Buffer.input, 0)
                chunk(rank, Buffer.input).reduce(c1, 0)
            distance *= 2
        # Broadcast tree - root is Rank 0
        distance = distance // 2
        while distance >= 1:
            # Copy to the right neighbor that is distance away
            for rank in range(0, size, distance*2):
                peer = rank + distance
                chunk(rank, Buffer.input, 0).copy(peer, Buffer.input, 0)
            distance = distance // 2

        # Mirrored version of the tree
        # Reduce tree - reducing onto Rank N-1
        if trees == 2:
            distance = 1
            while distance <= size // 2:
                # Reduce onto the right neighbor that is distance away
                for rank in range(size-1, 0, -distance*2):
                    peer = rank - distance
                    c1 = chunk(peer, Buffer.input, 1)
                    chunk(rank, Buffer.input, 1).reduce(c1)
                distance *= 2
            # Broadcast tree - root is Rank N-1
            distance = distance // 2
            while distance >= 1:
                # Copy to the left neighbor that is distance away
                for rank in range(size-1, 0, -distance*2):
                    peer = rank - distance
                    chunk(rank, Buffer.input, 1).copy(peer, Buffer.input, 1)
                distance = distance // 2

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('trees', type=int, choices=[1, 2], help ='number of trees')
parser.add_argument('instances', type=int, help ='number of instances')

parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allreduce_binomial_tree(args.num_gpus, args.instances, args.trees, args.protocol)