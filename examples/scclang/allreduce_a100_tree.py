# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Halving-doubling implementation of allreduce

import argparse

from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce


def allreduce(ways, instances):
    topology = fully_connected(8)
    size = topology.num_nodes() #  Number of gpus
    logical_chunk = 8 * ways
    tb_per_channel = 12
    collective = AllReduce(size, logical_chunk, True, "allreduce")
    with SCCLProgram("allreduce_a100_tree", topology, collective, instances, 'Simple', interleaved_replication=False):
        # 1 reduction between pairs of gpus of count
        def reduce_tree(pairs, count, next_index, lc, sendtb, recvtb):
            current_index = next_index.copy()
            for r in range(size):
                next = r ^ pairs
                offset = (count if r <= next else 0) 
                next_index[next] += offset
                # Split the reduce into two separate reduces to enable fused instructions
                block = 2 ** pairs
                for x in range(count):
                    index = current_index[r] + offset + lc*8 + x
                    c = chunk(Buffer.input, r, index)
                    c.reduce(next, Buffer.input, index, ch=lc, sendtb=sendtb, recvtb=recvtb)


        # Propagates reduced chunks in reverse order 
        def propagate_tree(pairs, count, next_index, lc, sendtb, recvtb):
            current_index = next_index.copy()            
            for r in range(size):
                next = r ^ pairs
                offset = (count if r > next else 0) 
                next_index[r] -= offset
                index = current_index[r] + lc*8
                c = chunk(Buffer.input, r, index, count)
                c.send(next, Buffer.input, index, ch=lc, sendtb=sendtb, recvtb=recvtb)

        next_index = [0] * 8
        reduce_tree(1, 4, next_index, 0, 0, 1)
        reduce_tree(2, 2, next_index, 0, 1, 2)
        reduce_tree(4, 1, next_index, 0, 2, 3)

        propagate_tree(4, 1, next_index, 0, 2, 3)
        propagate_tree(2, 2, next_index, 0, 1, 2)
        propagate_tree(1, 4, next_index, 0, 0, 1)

        if ways > 1:
            next_index = [0] * 8
            lc = 1
            reduce_tree(4, 4, next_index, lc, 8, 9)
            reduce_tree(2, 2, next_index, lc, 9, 10)
            reduce_tree(1, 1, next_index, lc, 10, 11)

            propagate_tree(1, 1, next_index, lc, 10, 11)
            propagate_tree(2, 2, next_index, lc, 9, 10)
            propagate_tree(4, 4, next_index, lc, 8, 9)
            
        if ways > 2:
            next_index = [0] * 8
            lc = 2
            reduce_tree(2, 4, next_index, lc, 4, 5)
            reduce_tree(1, 2, next_index, lc, 5, 6)
            reduce_tree(4, 1, next_index, lc, 6, 7)

            
            propagate_tree(4, 1, next_index, lc, 6, 7)
            propagate_tree(1, 2, next_index, lc, 5, 6)
            propagate_tree(2, 4, next_index, lc, 4, 5)
            

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('ways', type=int, help='number of parallel trees to perform reduction min:1 max:2')
parser.add_argument('instances', type=int, help='number of instances')
args = parser.parse_args()
assert args.ways >=0 and args.ways <= 3
allreduce(args.ways, args.instances)
