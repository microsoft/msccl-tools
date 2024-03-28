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

        for rank in range(size):
            for tb in range(size):
                index = rank * size
                c = chunk(rank, Buffer.input, index+tb)
                # step1 make sure the data is ready
                for nghr in range(size):
                    peer_index = nghr * size
                    if rank != nghr:
                        c_peer = chunk(rank, Buffer.input, peer_index+tb)
                        c_peer.signal(nghr, Buffer.input, peer_index+tb, sendtb=tb)
                for nghr in range(size):
                    if rank != nghr:
                        c.wait(nghr, Buffer.input, index+tb, recvtb=tb)
                # step2 reduce the chunks and send to peers
                for nghr in range(size):
                    if rank != nghr:
                        c.reduce_mscclpp(chunk(nghr, Buffer.input, index+tb), recvtb=tb)
                for nghr in range(size):
                    if rank != nghr:
                        c.put(nghr, Buffer.input, index, sendtb=tb)
                # step3 signal the peers to receive the chunks
                for nghr in range(size):
                    if rank != nghr:
                        c.signal(nghr, Buffer.input, index, sendtb=tb)
                for nghr in range(size):
                    if rank != nghr:
                        peer_index = nghr * size
                        c_peer = chunk(rank, Buffer.input, peer_index+tb)
                        c_peer.wait(nghr, Buffer.input, peer_index, recvtb=tb)

        Json()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple'], help='Protocol')

args = parser.parse_args()

allreduce_allpairs(args.num_gpus, args.instances, args.protocol)
