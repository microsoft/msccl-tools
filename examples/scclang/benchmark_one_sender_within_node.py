# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

# Performance test
# Within node:
# GPU 0 sends to 1, (1 and 2), (1, 2, and 3) .... or (1, 2, ..., 7)
# Across nodes:
# GPU 0 sends to 8, (8 and 9), (8, 9, and 10) .... or (8, 9, ..., 15)
def test(num_gpus, num_outgoing, instances):
    size = num_outgoing+1 
    topology = fully_connected(size)
    collective = AllReduce(size, 1, False)
    with SCCLProgram(f"gpu0_sendsto{num_outgoing}gpus_withinnode", topology, collective, instances):
        for _ in range(100):
            for i in range(num_outgoing):
                chunk(0, Buffer.input, 0).send(i+1, Buffer.output, 0)
        XML()


parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus within a node')
parser.add_argument('num_outgoing', type=int, help='number of unique gpus 0 is sending to')
parser.add_argument('instances', type=int, help ='number of instances')
args = parser.parse_args()


assert args.num_outgoing < args.num_gpus

test(args.num_gpus, args.num_outgoing, args.instances)
