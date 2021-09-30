# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import sccl
from sccl.autosynth.registry import register_synthesis_plan


def test_sccl_init(capsys):
    sccl.init(4, 'not_a_machine_type', ('alltoall', 0))
    out, err = capsys.readouterr()
    assert 'No plan found' in out
    sccl.init(2, 'ndv2', ('alltoall', '1MB'))
    out, err = capsys.readouterr()
    assert 'synthesize_ndv2_relay_alltoall' in out
    sccl.init(9, 'ndv4', ('alltoall', '1MB'))
    out, err = capsys.readouterr()
    assert 'synthesize_ndv4_hierarchical_alltoall' in out


def test_register_plan():
    @register_synthesis_plan('allgather', 'fancy_machine', sizes=(0, '4MB'))
    def dummy_plan(m, s):
        pass

    @register_synthesis_plan('allgather', ['m1', 'm2'], sizes=[(0, '4MB'), ('1GiB', None)])
    def dummy_plan(m, s):
        pass
