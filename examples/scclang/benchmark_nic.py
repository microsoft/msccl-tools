# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

# Performance stress test for NVSwitch
# Each GPU i in a node makes C connections and sends over those connections
# 1 connections: Node 0 GPU i sends to Node 1 GPU i
# 2 connections: Node 0 GPU i sends to Node 1 GPU i and Node 1 GPU i+1
# etc.
# Note: If C >= num_gpus we are using multiple NICs to generate traffic
def test(num_gpus, num_connections, instances, repeats):
    size = num_gpus*2
    # print("Size", size)
    # print("Connections", num_connections)
    topology = fully_connected(size)
    collective = AllReduce(size, size, False)
    with SCCLProgram("NICStressTest", topology, collective, instances):
        for it in range(repeats):
            for i in range(size):
                for c in range(num_connections):
                    j = (num_gpus + i + c) % size
                    # print(f"GPU{i}->GPU{j} chunk {j}")
                    chunk(i, Buffer.input, j).send(j, Buffer.output, i)
        XML()


parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus within a node')
parser.add_argument('num_connections', type=int, help='number of simultaneous cross-node connections each GPU in a node makes')
parser.add_argument('instances', type=int, help ='number of instances')
parser.add_argument('--repeats', type=int, default=100, help='number of repeated iterations')
args = parser.parse_args()


test(args.num_gpus, args.num_connections, args.instances, args.repeats)
