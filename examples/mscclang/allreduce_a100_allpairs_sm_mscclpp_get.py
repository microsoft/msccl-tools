# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def allreduce_allpairs(gpus, instances, protocol):
    size = gpus
    chunksperloop = gpus * gpus
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, True)
    with MSCCLProgram("allreduce_pairs", topology, collective, instances, protocol=protocol,
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual, dependence_nop=True):

        # Each rank sends the nth chunk to the nth rank into scratch space
        for rank in range(size):
            for tb in range(size):
                index = rank * size
                c = chunk(rank, Buffer.input, index + tb)
                # make sure the data is ready
                for nghr in range(size):
                    if rank != nghr:
                        c.signal(nghr, Buffer.input, index + tb, sendtb=tb)
                for nghr in range(size):
                    if rank != nghr:
                        c.wait(nghr, Buffer.input, index + tb, recvtb=tb)
                # reduce the chunks
                for nghr in range(size):
                    if rank != nghr:
                        c.reduce_mscclpp(chunk(nghr, Buffer.input, index + tb), recvtb=tb)
                for nghr in range(size):
                    if rank != nghr:
                        c.signal(nghr, Buffer.input, index, sendtb=tb)

        # wait for all the chunks is ready, then get the chunks
        for rank in range(size):
            for tb in range(size):
                for nghr in range(size):
                    if rank != nghr:
                        index = nghr * size
                        c = chunk(rank, Buffer.input, index + tb)
                        c.wait(nghr, Buffer.input, index, recvtb=tb)
                for nghr in range(size):
                    if rank != nghr:
                        c.get(nghr, Buffer.input, index, recvtb=tb)

        Json()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple'], help='Protocol')

args = parser.parse_args()

allreduce_allpairs(args.num_gpus, args.instances, args.protocol)
