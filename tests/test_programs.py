# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import msccl
from msccl.topologies import fully_connected
from msccl.language.collectives import *
import os
import pytest

def test_registered_alltoall_yifan():
    from msccl.programs.alltoall_a100_yifan import alltoall_hierarchical 

    num_nodes = 4
    gpus_per_node = 8
    num_ranks = num_nodes * gpus_per_node
    topology = fully_connected(num_ranks)
    collective = AllToAll(num_ranks, 1, inplace=False)
    with MSCCLProgram("hierarchical_all_to_all", topology, collective, 1):
        alltoall_hierarchical(num_nodes, gpus_per_node)
        assert Check()

def test_registered_alltoall_8kp1():
    from msccl.programs.alltoall_a100_8kp1 import alltoall_three_step 

    num_nodes = 9
    gpus_per_node = 8
    num_ranks = num_nodes * gpus_per_node
    topology = fully_connected(num_ranks)
    collective = AllToAll(num_ranks, 1, inplace=False)
    with MSCCLProgram("hierarchical_all_to_all", topology, collective, 1):
        alltoall_three_step(num_nodes, gpus_per_node)
        assert Check()
        XML()

def test_registered_allreduce_ring():
    from msccl.programs.allreduce_a100_ring import allreduce_ring 

    num_ranks = 8
    instances = 4
    topology = fully_connected(num_ranks)
    collective = AllReduce(num_ranks, num_ranks, inplace=True)
    with MSCCLProgram(f"allreduce_ring", topology, collective, instances,
        protocol="LL128", threadblock_policy=ThreadblockPolicy.manual):
        allreduce_ring(num_ranks, num_ranks)
        assert Check()
        XML()

def test_registered_allreduce_allpairs():
    from msccl.programs.allreduce_allpairs import allreduce_allpairs

    num_ranks = 8
    instances = 2
    topology = fully_connected(num_ranks)
    collective = AllReduce(num_ranks, num_ranks*num_ranks, inplace=True)
    with MSCCLProgram(f"allreduce_allpairs", topology, collective, instances,
        protocol="LL", threadblock_policy=ThreadblockPolicy.manual):
        allreduce_allpairs(num_ranks)
        assert Check()
        XML()

def test_registered_ndv4_allreduce(capsys):
    msccl.init('ndv4', 1, (msccl.Collective.allreduce, (512, 1024)))
    out, err = capsys.readouterr()
    assert 'ndv4_allpairs_allreduce_config1 with LL protocol' in out

    msccl.init('ndv4', 1, (msccl.Collective.allreduce, (82944, 458752)))
    out, err = capsys.readouterr()
    assert 'ndv4_allpairs_allreduce_config2 with LL protocol' in out

    msccl.init('ndv4', 1, (msccl.Collective.allreduce, (458752, 2129920)))
    out, err = capsys.readouterr()
    assert 'ndv4_ring_allreduce_config1 with LL protocol' in out

    msccl.init('ndv4', 1, (msccl.Collective.allreduce, (2129920, 22806528)))
    out, err = capsys.readouterr()
    assert 'ndv4_ring_allreduce_config2 with LL128 protocol' in out


def test_registered_ndv4_alltoall(capsys):
    msccl.init('ndv4', 8, (msccl.Collective.alltoall, ('1MB', '32MB')))
    out, err = capsys.readouterr()
    assert 'ndv4_alltoall_hierarchical_config1 with LL128 protocol' in out

    msccl.init('ndv4', 8, (msccl.Collective.alltoall, ('32MB', '64MB')))
    out, err = capsys.readouterr()
    assert 'ndv4_alltoall_hierarchical_config2 with Simple protocol' in out

    # msccl.init('ndv4', 64, (msccl.Collective.alltoall, ('32MB', '64MB')))
    # out, err = capsys.readouterr()
    # assert 'ndv4_alltoall_three_step with Simple protocol' in out
