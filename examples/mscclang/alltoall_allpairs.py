import argparse
import humanfriendly

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllToAll
from msccl.language.simulator import World, dgx2_top
# One-step AllToAll program
# Each gpu makes sends and receives a chunk from every other gpu

def alltoall(num_ranks, size, protocol):
    World.set_top(dgx2_top)
    topology = fully_connected(num_ranks)
    collective = AllToAll(num_ranks, 1, inplace=False)

    with MSCCLProgram("alltoall_allpairs", topology, collective, instances=-1, protocol=protocol, instr_fusion=False):
        for r in range(num_ranks):
            for index in range(num_ranks):
                chunk(r, Buffer.input, index).copy(index, Buffer.output, r)
        # Simulate()
        AutotuneBestSchedule(size, iterations=100) #, timing_from_pickle(size, open('single_buffer.pkl', 'rb')))
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
# parser.add_argument('instances', type=int, help ='number of instances')
parser.add_argument('size', type=str, help='size at which to simulate')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

alltoall(args.num_gpus, humanfriendly.parse_size(args.size), args.protocol)
