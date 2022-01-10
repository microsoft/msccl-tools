# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

def show():
    print()
    print(f"SCCL_CONFIG = {os.environ['SCCL_CONFIG']}")
    print(f"NCCL_MIN_NCHANNELS = {os.environ['NCCL_MIN_NCHANNELS']}")
    print(f"NCCL_NET_SHARED_BUFFERS = {os.environ['NCCL_NET_SHARED_BUFFERS']}")
    print(f"Contents of {os.environ['SCCL_CONFIG']}:")
    with open(os.environ['SCCL_CONFIG']) as f:
        print(f.read())
    print()


print('=== Trigger a builtin synthesis plan ===')

import sccl
sccl.init('ndv4', 9, (sccl.Collective.alltoall, '1GB'))

show()


print('=== Register additional plans from a library ===')

import sccl_presynth
sccl.init('ndv2', 3,
    (sccl.Collective.alltoall, '1GB'),
    (sccl.Collective.allgather, (128, '1KB')))

show()


print('=== Register custom plans ===')

from sccl.autosynth.registry import register_synthesis_plan

@register_synthesis_plan(sccl.Collective.alltoall, 'ndv9000', lambda m: m == 1, ('1MB', None))
def alltoall_9000(machines):
    return """<algo name="a2andv9000" nchunksperloop="2" nchannels="1" inplace="0" ngpus="2" proto="Simple">
    ...
    </algo>"""

sccl.init('ndv9000', 1, (sccl.Collective.alltoall, '2MB'))

show()


print('=== Overlapping size ranges ===')

register_synthesis_plan(sccl.Collective.alltoall, 'ndv9000', lambda m: m == 1, (0, '1KB'), protocol='LL')(alltoall_9000)
register_synthesis_plan(sccl.Collective.alltoall, 'ndv9000', lambda m: m == 1, ('1KB', '1MB'), protocol='LL128')(alltoall_9000)

sccl.init('ndv9000', 1, (sccl.Collective.alltoall, ('2KB', None)))

show()


print('=== SCCLang program ===')

from sccl.autosynth.registry import register_sccl_program
from sccl.topologies import line
from sccl.language import *

@register_sccl_program(line(2), 'allgather', 'two_gpus', machines= lambda m: m == 1)
def trivial_allgather(prog, nodes):
    chunk(Buffer.input, 0, 0).send(0, Buffer.output, 0).send(1)
    chunk(Buffer.input, 1, 0).send(1, Buffer.output, 1).send(0)

sccl.init('two_gpus', 1, (sccl.Collective.allgather, (0, None)))

show()


print('=== SCCLang program example ====')

from sccl.topologies import fully_connected
from sccl.programs.allreduce_a100_ring import allreduce_ring

@register_sccl_program(fully_connected(8), 'allreduce', 'ndv4', chunk_factor=8, inplace=True,
    instances=4, protocol='LL128', threadblock_policy=ThreadblockPolicy.manual, machines=lambda x: x == 1)
def ndv4_ring_allreduce(prog, nodes):
    allreduce_ring(size=8, channels=8)

sccl.init('ndv4', 1, (sccl.Collective.allreduce, (0, None)))


show()