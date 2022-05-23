import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllToAll

# One-step AllToAll program
# Each gpu makes sends and receives a chunk from every other gpu

def alltoall(num_ranks, instances, protocol):
    topology = fully_connected(num_ranks)
    collective = AllToAll(num_ranks, 1, inplace=False)

    with MSCCLProgram("alltoall_allpairs", topology, collective, instances=instances, protocol=protocol):
        for r in range(num_ranks):
            for index in range(num_ranks):
                chunk(r, Buffer.input, index).copy(index, Buffer.output, r)
        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help ='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

alltoall(args.num_gpus, args.instances, args.protocol)
