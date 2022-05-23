# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllGather

# Ring allgather for A100s
# Vary channels from [1-8] to divide parts of the ring over multiple channels/tbs.
# channels=1 is standard ring, all chunks are assigned to the same tb/channel
# channels=8 devotes 1 tb/channel to handling 1 chunk of the data
def allgather_ring(size, channels, instances, protocol):
    topology = fully_connected(size)
    collective = AllGather(size, 1, True)
    with MSCCLProgram(f"allgather_ring_{channels}channelsperring", topology, collective, instances,
         protocol=protocol, threadblock_policy=ThreadblockPolicy.manual):
        for step in range(0, size-1):
            for index in range(0, size):
                rank = (index + step) % size
                c = chunk(rank, Buffer.output, index)
                next_rank = (index + step + 1) % size
                channel = index%channels
                c = c.copy(next_rank, Buffer.output, index, sendtb=channel, recvtb=channel, ch=channel)   
        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('channels', type=int, help='Number of channels to use for 1 instance of the ring [1-8]')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL128', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

allgather_ring(args.num_gpus, args.channels, args.instances, args.protocol)
