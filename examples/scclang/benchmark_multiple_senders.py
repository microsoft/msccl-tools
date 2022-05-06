# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

# Performance test
# Within node:
# GPU 0 sends to 4, 1 sends to 5, 2 sends to 6, 3 sends to 7
def test(num_gpus, instances):
    size = num_gpus
    topology = fully_connected(size)
    collective = AllReduce(size, 1, False)
    with SCCLProgram(f"everyone_send_withinnode", topology, collective, instances):
        for _ in range(100):
            for i in range(0, size//2):
                chunk(i, Buffer.input, 0).send(i+size//2, Buffer.output, 0)
        XML()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus within a node')
parser.add_argument('instances', type=int, help ='number of instances')

args = parser.parse_args()

test(args.num_gpus, args.instances)
