# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .common import *
from msccl.topologies import fully_connected, ring, distributed_fully_connected
from msccl.collectives import alltoall
from msccl.instance import Instance
from msccl.path_encoding import PathEncoding
from msccl.distributors import *


def test_greedy_alltoall():
    num_nodes = 2
    num_copies = 2
    local_topo = fully_connected(num_nodes)
    encoding = PathEncoding(local_topo, alltoall(num_nodes))
    local_algo = encoding.solve(Instance(1))
    dist_topo = distributed_fully_connected(local_topo, num_copies, remote_bw=1)
    dist_algo = synthesize_greedy_distributed_alltoall(dist_topo, local_algo)
    dist_algo.check_implements(alltoall(num_nodes * num_copies))

def test_alltoall_subproblem():
    num_nodes = 2
    num_copies = 2
    local_topo = ring(num_nodes)
    sub_coll, sub_topo = make_alltoall_subproblem_collective_and_topology(local_topo, num_copies, [0])
    encoding = PathEncoding(sub_topo, sub_coll)
    sub_algo = encoding.solve(Instance(3, extra_rounds=1))
    dist_algo = synthesize_alltoall_subproblem(sub_algo, num_copies)
    dist_algo.check_implements(alltoall(num_nodes * num_copies))
