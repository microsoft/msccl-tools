# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce


def allreduce(ways, instances):
    topology = fully_connected(8)
    size = topology.num_nodes() #  Number of gpus
    logical_chunk = 8 * ways
    tb_per_channel = 12
    collective = AllReduce(size, instances*logical_chunk, True, "allreduce")
    with SCCLProgram("allreduce_a100", topology, collective, instances*logical_chunk, 'LL128'):

        # 1 reduction between pairs of gpus of count
        def reduce_tree(pairs, count, next_index, lc, i, sendtb, recvtb):
            current_index = next_index.copy()
            for r in range(size):
                next = r ^ pairs
                offset = (count if r <= next else 0) 
                next_index[next] += offset
                # Split the reduce into two separate reduces to enable an optimization
                # Hack to get RRS before RRC
                block = 2 ** pairs
                reverse = (r // block) % block
                for x in range(count):
                    # Uncomment to get RRS before RRC - but buggy :(
                    # if reverse == 0:
                    #     x = count - 1 - x
                    index = current_index[r] + offset + i*logical_chunk + lc*8 + x
                    c = Rank(r).input(index)
                    c.reduce(next, Buffer.input, index, ch=lc + i * ways, sendtb=sendtb+i*tb_per_channel, recvtb=recvtb+i*tb_per_channel)
                # Uncomment this and comment out L30 for loop to send/reduce with counts > 1
                # index = current_index[r] + offset + i*logical_chunk + lc*8 
                # c = Rank(r).input(index, count)
                # c.reduce(next, Buffer.input, index, ch=lc + i * ways, sendtb=sendtb+i*tb_per_channel, recvtb=recvtb+i*tb_per_channel)

        # Propagates reduced chunks in reverse order 
        def propagate_tree(pairs, count, next_index, lc, i, sendtb, recvtb):
            current_index = next_index.copy()            
            for r in range(size):
                next = r ^ pairs
                offset = (count if r > next else 0) 
                next_index[r] -= offset
                index = current_index[r] + i*logical_chunk + lc*8
                c = Rank(r).input(index, count)
                c.send(next, Buffer.input, index, ch=lc + i * ways, sendtb=sendtb+i*tb_per_channel, recvtb=recvtb+i*tb_per_channel)

        for i in range(instances):
            next_index = [0] * 8
            reduce_tree(1, 4, next_index, 0, i, 0, 1)
            reduce_tree(2, 2, next_index, 0, i, 1, 2)
            reduce_tree(4, 1, next_index, 0, i, 2, 3)

            propagate_tree(4, 1, next_index, 0, i, 2, 3)
            propagate_tree(2, 2, next_index, 0, i, 1, 2)
            propagate_tree(1, 4, next_index, 0, i, 0, 1)

            if ways > 1:
                next_index = [0] * 8
                lc = 1
                reduce_tree(2, 4, next_index, lc, i, 4, 5)
                reduce_tree(4, 2, next_index, lc, i, 5, 6)
                reduce_tree(1, 1, next_index, lc, i, 6, 7)

                
                propagate_tree(1, 1, next_index, lc, i, 6, 7)
                propagate_tree(4, 2, next_index, lc, i, 5, 6)
                propagate_tree(2, 4, next_index, lc, i, 4, 5)

            if ways > 2:
                next_index = [0] * 8
                lc = 2
                reduce_tree(4, 4, next_index, lc, i, 8, 9)
                reduce_tree(1, 2, next_index, lc, i, 9, 10)
                reduce_tree(2, 1, next_index, lc, i, 10, 11)

                propagate_tree(2, 1, next_index, lc, i, 10, 11)
                propagate_tree(1, 2, next_index, lc, i, 9, 10)
                propagate_tree(4, 4, next_index, lc, i, 8, 9)

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('ways', type=int, help='number of parallel trees to perform reduction min:1 max:3')
parser.add_argument('instances', type=int, help='number of instances')
args = parser.parse_args()
assert args.ways >=0 and args.ways <= 3
allreduce(args.ways, args.instances)
