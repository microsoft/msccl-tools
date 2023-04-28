import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def allpairs_reduce_scatter(gpuIds, size, offset):
    ngpus = len(gpuIds)

    # Each rank sends the nth chunk to the nth rank into scratch space
    for r1 in range(ngpus):
        for r2 in range(ngpus):
            if gpuIds[r1] != gpuIds[r2]:
                index = offset + r2 * size
                c = chunk(gpuIds[r1], Buffer.input, index, size=size)
                c.copy(gpuIds[r2], 'scratch', sendtb=gpuIds[r2], recvtb=gpuIds[r1])

    # Each rank performs a local reduction on the nth chunk
    # Utilize 8 threadblocks for this reduction for better parallelism
    for r in range(ngpus):
        for index in range(0, size * (ngpus-1)):
                c = chunk(gpuIds[r], Buffer.input, offset + r*size + (index % size))
                c.reduce(chunk(gpuIds[r], 'scratch', index), sendtb=(index % size))


def allpairs_all_gather(gpuIds, size, offset):
    ngpus = len(gpuIds)

    # Each rank sends its nth chunk to all other gpus
    for r1 in range(ngpus):
        for r2 in range(ngpus):
            if r1 != r2:
                index = offset + r1 * size
                c = chunk(gpuIds[r1], Buffer.input, index, size)
                c.copy(gpuIds[r2], Buffer.input, index, sendtb=gpuIds[r2], recvtb=gpuIds[r1])

# Performs two levels of allReduce
def hierarchical_allreduce(gpus, instances, protocol):
    ncols = 2
    nrows = gpus // ncols
    chunkperloop = gpus * gpus
    topology = fully_connected(gpus)
    collective = AllReduce(gpus, chunkperloop, True)

    with MSCCLProgram("hierarchical_allreduce", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual, dependence_nop=True):     
        
        # A 4 x 3 GPU arranagement: 4 local GPUs, 3 instances, GPU Ids are numbered as such
        # 0  4   8
        # 1  5   9
        # 2  6   10
        # 3  7  11
        # Reduce-Scatter on each column first, assumption being GPUs in a column have faster connectivity - NVLINK
        # Each GPU exchanges (nrows - 1) * 1/rows of data with other GPUs in the same column 
        # After this step, first GPU in each column will have 1st 1/nrows, 2nd GPU will have 2nd of 1/nrows data reduced
        size = chunkperloop // nrows
        offset = 0
        for n in range(ncols):
            gpuIds = []
            for m in range(nrows): # collect all GPU Ids in a column
                gpuIds.append( n * nrows + m)
            
            allpairs_reduce_scatter(gpuIds, size, 0)

        # Reduce-Scatter across rows, assumption being GPUs in a row have slower connectivity - PCIe, IP NW
        # Each GPU exachanges (1 / rows * cols) * (cols - 1) of data with other GPUs in the same row - less data is exchanged
        # After this step, first GPU each row, will have 1st 1/(nrows * ncols), 2nd will have 2nd of 1/(nrows * ncols)
        offset = size
        size = chunkperloop // (nrows * ncols)
        for n in range(nrows):
            gpuIds = []
            for m in range(ncols):
                gpuIds.append(n + m * nrows)

            allpairs_reduce_scatter(gpuIds, size, offset * n)

        # AllGather: AllGather phase goes in reverse order, first gather across rows of GPU
        # After this step, Each GPU in a rows have 1/ncols of data
        for n in range(nrows):
            gpuIds = []
            for m in range(ncols):
                gpuIds.append(n + m * nrows)

            allpairs_all_gather(gpuIds, size, offset * n)

        # AllGather: AllGather phase goes in reverse order, 2nd AllGather across columns of GPU
        # After this step, Each GPU the systems will have complete reduced data
        size = chunkperloop // nrows
        offset = 0
        for n in range(ncols):
            gpuIds = []
            for m in range(nrows):
                gpuIds.append( n * nrows + m)

            allpairs_all_gather(gpuIds, size, 0)

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL128', 'LL'], help='Protocol')

args = parser.parse_args()

hierarchical_allreduce(args.num_gpus, args.instances, args.protocol)