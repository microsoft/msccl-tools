# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import sccl
from sccl.autosynth.registry import register_synthesis_plan


def test_sccl_init():
    sccl.init(4, 'not_a_machine_type', ('alltoall', 0))
    sccl.init(2, 'dgx1', ('alltoall', '1MB'))


def test_register_plan():
    @register_synthesis_plan('allgather', 'fancy_machine', sizes=(0, '4MB'))
    def dummy_plan(m, s):
        pass

    @register_synthesis_plan('allgather', ['m1', 'm2'], sizes=[(0, '4MB'), ('1GiB', None)])
    def dummy_plan(m, s):
        pass
