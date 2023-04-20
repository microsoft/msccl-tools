# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Halving-doubling implementation of allreduce

import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce


def allreduce(ways, instances, protocol):
    topology = fully_connected(4)
    size = topology.num_nodes() #  Number of gpus
    logical_chunk = 8 * ways
    collective = AllReduce(size, logical_chunk, True)
    with MSCCLProgram("allreduce_a100_recursive_doubling_halving", topology, collective, instances, protocol, interleaved_replication=False):
        # 1 reduction between pairs of gpus of count
        def recursive_doubling(pairs, count, next_index, lc, sendtb, recvtb):
            current_index = next_index.copy()
            for r in range(size):
                next = r ^ pairs
                offset = (count if r <= next else 0) 
                next_index[next] += offset
                # Split the reduce into two separate reduces to enable fused instructions
                block = 2 ** pairs
                for x in range(count):
                    index = current_index[r] + offset + lc*8 + x
                    c1 = chunk(r, Buffer.input, index)
                    c = chunk(next, Buffer.input, index)
                    c.reduce(c1, sendtb=sendtb, recvtb=recvtb)


        # Propagates reduced chunks in reverse order 
        def recursive_halving(pairs, count, next_index, lc, sendtb, recvtb):
            current_index = next_index.copy()            
            for r in range(size):
                next = r ^ pairs
                offset = (count if r > next else 0) 
                next_index[r] -= offset
                index = current_index[r] + lc*8
                c = chunk(r, Buffer.input, index, count)
                c.copy(next, Buffer.input, index, ch=lc, sendtb=sendtb, recvtb=recvtb)

        next_index = [0] * 8
        recursive_doubling(1, 4, next_index, 0, 0, 1)
        recursive_doubling(2, 2, next_index, 0, 1, 2)
        recursive_doubling(4, 1, next_index, 0, 2, 3)

        recursive_halving(4, 1, next_index, 0, 2, 3)
        recursive_halving(2, 2, next_index, 0, 1, 2)
        recursive_halving(1, 4, next_index, 0, 0, 1)

        if ways > 1:
            next_index = [0] * 8
            lc = 1
            recursive_doubling(4, 4, next_index, lc, 8, 9)
            recursive_doubling(2, 2, next_index, lc, 9, 10)
            recursive_doubling(1, 1, next_index, lc, 10, 11)

            recursive_halving(1, 1, next_index, lc, 10, 11)
            recursive_halving(2, 2, next_index, lc, 9, 10)
            recursive_halving(4, 4, next_index, lc, 8, 9)
            
        if ways > 2:
            next_index = [0] * 8
            lc = 2
            recursive_doubling(2, 4, next_index, lc, 4, 5)
            recursive_doubling(1, 2, next_index, lc, 5, 6)
            recursive_doubling(4, 1, next_index, lc, 6, 7)

            
            recursive_halving(4, 1, next_index, lc, 6, 7)
            recursive_halving(1, 2, next_index, lc, 5, 6)
            recursive_halving(2, 4, next_index, lc, 4, 5)
            

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('ways', type=int, help='number of parallel trees to perform reduction min:1 max:2')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
assert args.ways >= 1 and args.ways <= 3
allreduce(args.ways, args.instances, args.protocol)
