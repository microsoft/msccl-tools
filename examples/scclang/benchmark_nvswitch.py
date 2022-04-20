# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

# Performance stress test for NVSwitch
# Each GPU i in a node makes C connections and sends over those connections
# 1 connections: GPU i sends to GPU (i+1)%N
# 2 connections: GPU i sends to GPU (i+1)%N and GPU(i+2)%N
# etc.
# Note: If C >= num_gpus we are using multiple NICs to generate traffic
def test(num_gpus, num_connections, instances, repeats):
    size = max(num_gpus, num_connections+1)
    # print("Size", size)
    # print("Connections", num_connections)
    topology = fully_connected(size)
    collective = AllReduce(size, size*repeats, False)
    with SCCLProgram("NVSwitchStressTest", topology, collective, instances):
        for it in range(repeats):
            for i in range(num_gpus):
                for c in range(num_connections):
                    j = (i + c + 1) % size
                    src_idx = j*repeats + it
                    dst_idx = i*repeats + it
                    # print(f"GPU{i}->GPU{j} chunk {j}")
                    chunk(i, Buffer.input, src_idx).send(j, Buffer.output, dst_idx)
        XML()


parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus within a node')
parser.add_argument('num_connections', type=int, help='number of simultaneous connections each GPU in a node makes')
parser.add_argument('instances', type=int, help ='number of instances')
parser.add_argument('--repeats', type=int, default=100, help='number of repeated iterations')
args = parser.parse_args()


test(args.num_gpus, args.num_connections, args.instances, args.repeats)
