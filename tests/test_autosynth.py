# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import msccl
import os
from msccl.autosynth.registry import register_synthesis_plan


def test_msccl_init(capsys):
    msccl.init('not_a_machine_type', 4, ('alltoall', 0))
    out, err = capsys.readouterr()
    assert 'No plan found' in out
    assert not 'MSCCL_CONFIG' in os.environ
    assert 'NCCL_ALGO' not in os.environ

    msccl.init('ndv2', 2, ('alltoall', '1MB'))
    out, err = capsys.readouterr()
    assert 'synthesize_ndv2_relay_alltoall' in out
    assert 'MSCCL_CONFIG' in os.environ
    assert 'NCCL_IB_AR_THRESHOLD' not in os.environ
    assert 'NCCL_ALGO' in os.environ and os.environ['NCCL_ALGO'] == 'MSCCL,RING,TREE'

    os.environ['NCCL_ALGO'] = 'RING,FAKE_MSCCL'
    msccl.init('ndv4', 8, (msccl.Collective.alltoall, '2MB'))
    out, err = capsys.readouterr()
    assert 'ndv4_alltoall' in out
    assert 'NCCL_IB_AR_THRESHOLD' in os.environ
    assert 'NCCL_ALGO' in os.environ and os.environ['NCCL_ALGO'] == 'MSCCL,RING,FAKE_MSCCL'

    os.environ['NCCL_ALGO'] = 'HELLO,MSCCL,WORLD'
    msccl.init('ndv4', 16, (msccl.Collective.alltoall, '35MB'))
    out, err = capsys.readouterr()
    assert 'ndv4_alltoall' in out
    assert 'NCCL_ALGO' in os.environ and os.environ['NCCL_ALGO'] == 'HELLO,MSCCL,WORLD'


def test_register_plan():
    @register_synthesis_plan('allgather', 'fancy_machine', sizes=(0, '4MB'))
    def dummy_plan(m, s):
        pass

    @register_synthesis_plan('allgather', ['m1', 'm2'], sizes=[(0, '4MB'), ('1GiB', None)])
    def dummy_plan(m, s):
        pass
