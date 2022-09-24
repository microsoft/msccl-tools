# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.topologies import *
from msccl.collectives import *
from msccl.serialization import *

import os
import sys
import tempfile
import shutil

class in_tempdir:
    '''Context manager for changing to a temporary directory.'''
    def __init__(self):
        self.tempdir = tempfile.mkdtemp()

    def __enter__(self):
        self.cwd = os.getcwd()
        os.chdir(self.tempdir)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.cwd)
        shutil.rmtree(self.tempdir)

def _check_ncclizes(path):
    assert 0 == os.system(f'msccl ncclize {path} -o ncclized.msccl.xml')
    assert os.path.exists('ncclized.msccl.xml')

def test_run_as_module():
    assert 0 == os.system(f'{sys.executable} -m msccl --help')

def test_entrypoint():
    assert 0 == os.system('msccl --help')

def test_solve_instance():
    with in_tempdir():
        assert 0 == os.system('msccl solve instance Ring Allgather --nodes 4 --steps 1 -o algo.json')
        assert not os.path.exists('algo.json')
        assert 0 == os.system('msccl solve instance Ring Allgather --nodes 2 --steps 1 -o algo.json')
        assert os.path.exists('algo.json')
        assert 0 == os.system('msccl solve instance Ring Allgather --nodes 2 --steps 1 -o algo.json --force')
        _check_ncclizes('algo.json')

def test_extra_memory():
    with in_tempdir():
        assert 0 == os.system('msccl solve instance Ring -n 4 Alltoall -s 2 --extra-memory 0 -o algo.json')
        _check_ncclizes('algo.json')

def test_solve_least_steps():
    assert 0 == os.system('msccl solve least-steps Ring Allgather --nodes 2')
    assert 0 == os.system('msccl solve least-steps Ring Allgather --nodes 2 --initial-steps 2')

def test_solve_pareto_optimal():
    with in_tempdir():
        assert 0 == os.system('msccl solve pareto-optimal Ring Allgather --nodes 4 -d .')
        assert len(os.listdir('.')) == 1
    with in_tempdir():
        assert 0 == os.system('msccl solve pareto-optimal Ring Allgather --nodes 4 -d . --save-eagerly')
        assert len(os.listdir('.')) == 2
    assert 0 == os.system('msccl solve pareto-optimal Ring Alltoall --nodes 2 --assume-rpc-bound 1/1')
    assert 0 == os.system('msccl solve pareto-optimal Ring Alltoall --nodes 2 --no-monotonic-feasibility')

def test_ncclize():
    with in_tempdir():
        assert 0 == os.system('msccl solve instance Ring Allgather --nodes 2 --steps 1 -o algo.json')
        assert os.path.exists('algo.json')
        assert 0 == os.system('msccl ncclize algo.json -o ncclized1.msccl.xml')
        assert os.path.exists('ncclized1.msccl.xml')
        assert 0 == os.system('msccl ncclize algo.json -f --channel-policy One')
        assert 0 == os.system('msccl ncclize algo.json -f --channel-policy MatchTopology')
        assert 0 == os.system('msccl ncclize algo.json -f --no-merge-contiguous')
        assert 0 == os.system('msccl solve instance Star Alltoall --nodes 4 --steps 2 --rounds 4 -o algo_scratch.json')
        assert 0 == os.system('msccl ncclize algo_scratch.json -f --remap-scratch')
        assert 0 == os.system('msccl ncclize algo_scratch.json -f --greedy-scratch-sorting')

def test_custom_topology_and_collective():
    with in_tempdir():
        topo = Topology('CT', [[0, 1], [1, 0]])
        coll = build_collective('CC', 2, 1, lambda r, c: r == 0, lambda r, c: r == 1)
        save_msccl_object(topo, 'topo.json')
        save_msccl_object(coll, 'coll.json')
        assert 0 == os.system('msccl solve instance custom custom --topology-file topo.json --collective-file coll.json -s 1')

def test_solve_bound_rounds():
    assert '7/6' in os.popen('msccl analyze rounds DGX1 Allgather').read()

def test_find_isomorphisms():
    assert 0 == os.system('msccl analyze isomorphisms DGX1 DGX1')

def test_distribute_alltoall_greedy():
    with in_tempdir():
        assert 0 == os.system('msccl solve instance Ring Alltoall --nodes 4 --steps 2 -o local.json')
        assert 0 == os.system('msccl distribute alltoall-greedy local.json DistributedFullyConnected --copies 3 -o dist.json')
        assert os.path.exists('dist.json')
        _check_ncclizes('dist.json')
        assert 0 == os.system('msccl distribute alltoall-greedy local.json DistributedHubAndSpoke --nodes 8')
        assert 0 != os.system('msccl distribute alltoall-greedy local.json DistributedHubAndSpoke --nodes 5')

def test_distribute_alltoall_scatter_gather():
    with in_tempdir():
        assert 0 == os.system('msccl solve instance DGX1 Gather --root 5 --steps 2 -o gather.json')
        assert 0 == os.system('msccl solve instance DGX1 Scatter --root 5 --steps 2 -o scatter.json')
        assert 0 == os.system('msccl distribute alltoall-gather-scatter gather.json scatter.json --copies 2 -o alltoall.json')
        assert os.path.exists('alltoall.json')
        _check_ncclizes('alltoall.json')

def test_distribute_alltoall_scatter_gather_multiroot():
    with in_tempdir():
        assert 0 == os.system('msccl solve instance Ring -n 3 MultirootGather --roots 0 1 --steps 1 -o gather.json')
        assert 0 == os.system('msccl solve instance Ring -n 3 MultirootScatter --roots 1 2 --steps 1 -o scatter.json')
        assert 0 == os.system('msccl distribute alltoall-gather-scatter gather.json scatter.json --copies 2 -o alltoall.json')
        assert os.path.exists('alltoall.json')
        _check_ncclizes('alltoall.json')

def test_distribute_alltoall_subproblem():
    # TODO: make this test less brittle. Currentl it will break when algorithm naming is changed, but we don't actually
    # want to test for that.
    with in_tempdir():
        assert 0 == os.system('msccl distribute alltoall-create-subproblem Line -n 2 --copies 2')
        coll_name = 'AlltoallSubproblem.n2.copies2.msccl.json'
        topo_name = 'Subtopo.localLine.n2.relays.0.msccl.json'
        assert os.path.exists(coll_name)
        assert os.path.exists(topo_name)
        assert 0 == os.system('msccl solve instance custom custom --topology-file Subtopo.localLine.n2.relays.0.msccl.json --collective-file AlltoallSubproblem.n2.copies2.msccl.json -s 3 -r 4 -o subalgo.json')
        assert 0 == os.system('msccl distribute alltoall-stitch-subproblem subalgo.json --copies 2 -o stitched.json')
        assert os.path.exists('stitched.json')
        _check_ncclizes('stitched.json')

def test_compose_allreduce():
    with in_tempdir():
        assert 0 == os.system('msccl solve instance DGX1 ReduceScatter -s 2 -r 3 -c 2 -o reducescatter.json')
        assert 0 == os.system('msccl solve instance DGX1 Allgather -s 2 -r 3 -c 2 -o allgather.json')
        assert 0 == os.system('msccl compose allreduce reducescatter.json allgather.json -o allreduce.json')
        assert os.path.exists('allreduce.json')
        _check_ncclizes('allreduce.json')