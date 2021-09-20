# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

tb_per_channel = 10
def allreduce(instances):
    topology = fully_connected(8)
    size = topology.num_nodes() #  Number of gpus
    logical_chunk = 8
    collective = AllReduce(size, instances*logical_chunk, True, "allreduce")
    with SCCLProgram("allreduce_ndv2", topology, collective, instances*logical_chunk):

        # 1 reduction between pairs number of gpus 
        # Number of chunks reduced is depdendent on number of pairs
        # 1 pair => 4 chunks, 2 pairs => 2 chunks
        def reduce_ring(pairs, next_index, i, sendtb, recvtb):
            count = 8 // (2*pairs)
            current_index = next_index.copy()
            for r in range(size):
                next = r ^ pairs
                offset = (count if r < next else 0) 
                next_index[next] += offset
                # Split the reduce into two separate reduces to enable an optimization
                for x in range(count):
                    index = current_index[r] + offset + i * logical_chunk + x
                    c = Rank(r).input(index)
                    c.reduce(next, Buffer.input, index, ch=i*2, sendtb=sendtb+i*tb_per_channel, recvtb=recvtb+i*tb_per_channel)
                

        # Propagates reduced chunks in reverse order 
        def propagate_ring(pairs, next_index, i, sendtb, recvtb):
            count = 8 // (2*pairs)
            current_index = next_index.copy()
            for r in range(size):
                next = r ^ pairs
                offset = (count if r > next else 0) 
                next_index[r] -= offset
                index = current_index[r] + i*logical_chunk
                c = Rank(r).input(index, count)
                c.send(next, Buffer.input, index, ch=i*2+1, sendtb=sendtb+i*tb_per_channel, recvtb=recvtb+i*tb_per_channel)

        for i in range(instances):
            next_index = [0] * 8

            reduce_ring(1, next_index, i, 0, 1)
            reduce_ring(2, next_index, i, 1, 2)
            reduce_ring(4, next_index, i, 2, 3)

            propagate_ring(4, next_index, i, 4, 5)
            propagate_ring(2, next_index, i, 6, 7)
            propagate_ring(1, next_index, i, 8, 9)

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('instances', type=int, help='number of instances')
args = parser.parse_args()
allreduce(args.instances)
