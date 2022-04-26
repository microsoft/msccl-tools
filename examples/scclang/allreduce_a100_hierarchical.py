# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

def allreduce(num_local_gpus, num_nodes, instances):
    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    collective = AllReduce(num_gpus, num_local_gpus, True)

    def rank(n, g):
        return n * num_local_gpus + (g % num_local_gpus)

    with SCCLProgram("allreduce_2node_a100", topology, collective, instances):

        # Ring Reduce Scatter within each node
        # for n in range(num_nodes):
        #     for ch in range(0, num_local_gpus):
        #         for step in range(0, num_local_gpus-1):
        #             c = chunk(rank(n, ch+step+1), Buffer.input, ch).reduce(rank(n, ch+step+2), Buffer.input, ch)

        # Allpairs Reduce Scatter within each node
        for n in range(num_nodes):
            for g in range(0, num_local_gpus):
                for ch in range(0, num_local_gpus):
                    if ch != g:
                        chunk(rank(n,g), Buffer.input, ch).reduce(rank(n,ch), Buffer.input, ch)
        
        # Exchange across IBs
        for ch in range(0, num_local_gpus):
            chunk(rank(0, ch), Buffer.input, ch).rexchange(rank(1, ch), Buffer.input, ch)

        # Ring All gather within each node
        for n in range(num_nodes):
            for ch in range(0, num_local_gpus):
                for step in range(0, num_local_gpus-1):
                    chunk(rank(n, ch+step), Buffer.input, ch).send(rank(n, ch+step+1), Buffer.input, ch)
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--rs', type='str', default='ring', choices=['ring', 'allpairs'], help='Reduce scatter algorithm')
args = parser.parse_args()

assert args.num_nodes == 2, "Only works for 2 nodes right now"

allreduce(8, args.num_nodes, args.instances)
