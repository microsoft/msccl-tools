import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllGather

def allpairs_all_gather(gpuIds, size, offset):
    ngpus = len(gpuIds)

    # Each rank sends its nth chunk to all other gpus
    for r1 in range(ngpus):
        for r2 in range(ngpus):
            if r1 != r2:
                index = offset
                for r in range(size):   # one chunk per copy command, so they can be overlapped by the runtime
                    c = chunk(gpuIds[r1], Buffer.input, index, 1)
                    c.copy(gpuIds[r2], Buffer.input, index, sendtb=r2, recvtb=r1)
                    index += 1


# Performs two levels of allGather
def hierarchical_allgather(gpus, instances, protocol):
    ncols = 2
    nrows = gpus // ncols
    chunks_per_gpu = 1
    nchunks = gpus * chunks_per_gpu
    topology = fully_connected(gpus)

    collective = AllGather(gpus, chunks_per_gpu, True)

    with MSCCLProgram("hierarchical_allgather", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False, dependence_nop=True):
        
        # A100-PCIe arrangemment:
        # 0     1
        # 2     3
        #
        # A 4 x 3 GPU arranagement: 4 local GPUs, 3 instances, GPU Ids are numbered as such
        # 0  1   2
        # 3  4   5
        # 6  7   8
        # 9  10  11
        # AllGather: AllGather phase goes in reverse order, first gather across rows of GPU
        # Each GPU sends  1/(nrows * ncols) of data to all other GPUs in the row
        # After this step, Each GPU in a rows have 1/ncols of data
        size = nchunks // (nrows * ncols)        
        
        for n in range(nrows):
            gpuIds = []
            for m in range(ncols):
                gpuIds.append(n * ncols + m)

            allpairs_all_gather(gpuIds, size, offset=0)

        # AllGather: AllGather phase goes in reverse order, 2nd AllGather across columns of GPU
        # After this step, Each GPU the systems will have complete reduced data
        size = nchunks // nrows
        base = 0

        for n in range(ncols):
            gpuIds = []
            for m in range(nrows):
                gpuIds.append( n + m * ncols)

            allpairs_all_gather(gpuIds, size, offset= -1 * n * chunks_per_gpu)

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL128', 'LL'], help='Protocol')

args = parser.parse_args()

hierarchical_allgather(args.num_gpus, args.instances, args.protocol)