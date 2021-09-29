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
    collective = AllReduce(size, instances*logical_chunk, True, "allreduce")
    with SCCLProgram("allreduce_ndv2", topology, collective, instances*logical_chunk):
        # local reduce_scatter
        for lc in range(num_local_gpus//2):
            for r in range(num_local_gpus):
                if lc == (r % (num_local_gpus//2)):
                    continue
                within_socket_nghr = lc + (4 if (r >= num_local_gpus//2) else 0)
                for ch in range(instances):
                    index = (lc * instances+ch)*2
                    c = Rank(r).input(index, 2)
                    # reduce (rank, buffer, index)
                    c.reduce(within_socket_nghr, buffer=Buffer.input, index=index, ch=ch)
                    # d = d.reduce(within_socket_nghr, buffer=Buffer.input, index=index, ch=ch)
        #  cross-socket reduce_scatter
        for r in range(num_local_gpus):
            for ch in range(instances):
                index = ((r % (num_local_gpus//2))*instances + ch) * 2
                if r >= num_local_gpus // 2:
                    index += 1 # Handle the odd chunk
                lc = Rank(r).input(index)
                lc = lc.reduce((r+num_local_gpus//2) % num_local_gpus, buffer=Buffer.input, index=index, ch=ch)
                lc.send(r, Buffer.input, index, ch=ch*instances+1) # Reduce and send should be on different tbs
        #  # local all_gather
        for r in range(num_local_gpus):
            for ch in range(instances):
                index = ((r % (num_local_gpus//2))*instances + ch) * 2
                lc = Rank(r).input(index, 2)
                for t in range(num_local_gpus//2):
                    local_nghr = t + (num_local_gpus//2 if (r >= num_local_gpus//2) else 0)
                    if local_nghr == r:
                        continue
                    lc.send(local_nghr, buffer=Buffer.input, index=index, ch=ch)
        XML()
        # Check()

parser = argparse.ArgumentParser()
parser.add_argument('instances', type=int, help='number of instances')
args = parser.parse_args()
allreduce(args.instances)
