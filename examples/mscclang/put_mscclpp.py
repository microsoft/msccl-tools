# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def allreduce_allpairs(gpus, instances, protocol):
    size = gpus
    chunksperloop = gpus
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, True)
    with MSCCLProgram("allreduce_pairs", topology, collective, instances, protocol=protocol,
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual, dependence_nop=True):

        c = chunk(0, Buffer.input, 0, size=1)
        c.put(1, Buffer.input, index=0, sendtb=0)
        c.put(1, Buffer.input, index=1, sendtb=0)
        c.signal(1, Buffer.input, index=0, sendtb=0)
        c.signal(1, Buffer.input, index=1, sendtb=0)

        dc0 = chunk(1, Buffer.input, 1, size=1)
        dc1 = chunk(1, Buffer.input, 0, size=1)
        dc0.wait(0, Buffer.input, index=0, recvtb=1)
        dc1.wait(0, Buffer.input, index=1, recvtb=1)

        Json()
        #Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL128', 'LL'], help='Protocol')

args = parser.parse_args()

allreduce_allpairs(args.num_gpus, args.instances, args.protocol)
