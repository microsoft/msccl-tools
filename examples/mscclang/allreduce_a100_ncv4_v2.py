# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def tree_algo(tree, chnk, size):
    for i in range(size-1):
        nextNghr = tree[i+1]
        curNode = tree[i]
        c = chunk(nextNghr, Buffer.input, chnk)
        c.reduce(chunk(curNode, Buffer.input, chnk), sendtb=2*chnk, recvtb=2*chnk, ch=chnk)
    for i in range(size-1):
        curNode = tree[size-1-i]
        nextNghr = tree[size-1-i-1]
        c = chunk(curNode, Buffer.input, chnk)
        c.copy(nextNghr, Buffer.input, chnk, sendtb=2*chnk+1, recvtb=2*chnk+1, ch=chnk)

def allreduce_allpairs(gpus, instances, protocol):
    size = gpus
    chunksperloop = 2
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, True)
    with MSCCLProgram("allreduce_ncv4", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual, dependence_nop=True):
        tree_algo([3,2,1,0], 0, size)
        tree_algo([2,3,0,1], 1, size)

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL128', 'LL'], help='Protocol')

args = parser.parse_args()

allreduce_allpairs(args.num_gpus, args.instances, args.protocol)