# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import sccl
import os
from sccl.autosynth.registry import register_synthesis_plan


def test_sccl_init(capsys):
    sccl.init('not_a_machine_type', 4, ('alltoall', 0))
    out, err = capsys.readouterr()
    assert 'No plan found' in out
    assert not 'SCCL_CONFIG' in os.environ
    assert 'NCCL_ALGO' not in os.environ

    sccl.init('ndv2', 2, ('alltoall', '1MB'))
    out, err = capsys.readouterr()
    assert 'synthesize_ndv2_relay_alltoall' in out
    assert 'SCCL_CONFIG' in os.environ
    assert 'NCCL_IB_AR_THRESHOLD' not in os.environ
    assert 'NCCL_ALGO' in os.environ and os.environ['NCCL_ALGO'] == 'SCCL,RING,TREE'

    os.environ['NCCL_ALGO'] = 'RING,FAKE_SCCL'
    sccl.init('ndv4', 8, (sccl.Collective.alltoall, '2MB'))
    out, err = capsys.readouterr()
    assert 'ndv4_alltoall' in out
    assert 'NCCL_IB_AR_THRESHOLD' in os.environ
    assert 'NCCL_ALGO' in os.environ and os.environ['NCCL_ALGO'] == 'SCCL,RING,FAKE_SCCL'

    os.environ['NCCL_ALGO'] = 'HELLO,SCCL,WORLD'
    sccl.init('ndv4', 16, (sccl.Collective.alltoall, '35MB'))
    out, err = capsys.readouterr()
    assert 'ndv4_alltoall' in out
    assert 'NCCL_ALGO' in os.environ and os.environ['NCCL_ALGO'] == 'HELLO,SCCL,WORLD'


def test_register_plan():
    @register_synthesis_plan('allgather', 'fancy_machine', sizes=(0, '4MB'))
    def dummy_plan(m, s):
        pass

    @register_synthesis_plan('allgather', ['m1', 'm2'], sizes=[(0, '4MB'), ('1GiB', None)])
    def dummy_plan(m, s):
        pass
