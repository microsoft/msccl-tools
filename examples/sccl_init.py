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