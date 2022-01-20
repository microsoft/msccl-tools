# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from sccl.language import *
from sccl.topologies.distributed import *
from sccl.topologies.nvidia import *
from sccl.language.collectives import AllReduce

def allreduce(instances):
    topology = dgx1()
    num_local_gpus = 8
    size = topology.num_nodes() #  Number of gpus
    logical_chunk = 8
    collective = AllReduce(size, logical_chunk, True)
    with SCCLProgram("allreduce_ndv2", topology, collective, instances, interleaved_replication=False):
        # local reduce_scatter
        instances = 1
        for lc in range(num_local_gpus//2):
            for r in range(num_local_gpus):
                if lc == (r % (num_local_gpus//2)):
                    continue
                within_socket_nghr = lc + (4 if (r >= num_local_gpus//2) else 0)
                index = lc * 2
                c = chunk(Buffer.input, r, index, 2)
                c.reduce(within_socket_nghr, buffer=Buffer.input, index=index)
        #  cross-socket reduce_scatter
        for r in range(num_local_gpus):
            index = (r % (num_local_gpus//2)) * 2
            if r >= num_local_gpus // 2:
                index += 1 # Handle the odd chunk
            lc = chunk(Buffer.input, r, index)
            lc = lc.reduce((r+num_local_gpus//2) % num_local_gpus, buffer=Buffer.input, index=index)
            lc.send(r, Buffer.input, index, ch=1) # Reduce and send should be on different tbs
        #  local all_gather
        for r in range(num_local_gpus):
            index = (r % (num_local_gpus//2)) * 2
            lc = chunk(Buffer.input, r, index, 2)
            for t in range(num_local_gpus//2):
                local_nghr = t + (num_local_gpus//2 if (r >= num_local_gpus//2) else 0)
                if local_nghr == r:
                    continue
                lc.send(local_nghr, buffer=Buffer.input, index=index)
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('instances', type=int, help='number of instances')
args = parser.parse_args()
allreduce(args.instances)
