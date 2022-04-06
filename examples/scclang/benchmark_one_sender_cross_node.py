# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

# Performance test
# GPU 0 sends to 8, (8 and 9), (8, 9, and 10) .... or (8, 9, ..., 15)
def test(num_gpus, num_outgoing, instances):
    size = num_gpus + num_outgoing
    topology = fully_connected(size)
    collective = AllReduce(size, 1, False)
    with SCCLProgram(f"gpu0_sendsto{num_outgoing}gpus_crossnode", topology, collective, instances):
        for _ in range(100):
            for i in range(num_outgoing):
                peer = i+num_gpus
                chunk(0, Buffer.input, 0).send(peer, Buffer.output, 0)
        # dummy
        for i in range(1, 8):
            chunk(i, Buffer.input, 0).send(i, Buffer.output, 0)

        XML()
    
  

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus within a node')
parser.add_argument('num_outgoing', type=int, help='number of unique gpus 0 is sending to')
parser.add_argument('instances', type=int, help ='number of instances')

args = parser.parse_args()

assert args.num_outgoing <= args.num_gpus

test(args.num_gpus, args.num_outgoing, args.instances)
