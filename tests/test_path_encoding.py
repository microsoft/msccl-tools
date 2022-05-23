# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.path_encoding import PathEncoding
from msccl.topologies import fully_connected, line, dgx1
from msccl.collectives import *
from msccl.instance import Instance

def test_fc_noncombining():
    num_nodes = 2
    enc = PathEncoding(fully_connected(num_nodes), allgather(num_nodes))
    assert enc.solve(Instance(1, chunks=2)) == None
    assert enc.solve(Instance(2, chunks=2)) != None

def test_fc_combining_reducible():
    num_nodes = 2
    enc = PathEncoding(fully_connected(num_nodes), reduce_scatter(num_nodes))
    assert enc.solve(Instance(1, chunks=2)) == None
    assert enc.solve(Instance(2, chunks=2)) != None

def test_fc_combining_nonreducible():
    num_nodes = 2
    enc = PathEncoding(fully_connected(num_nodes), allreduce(num_nodes))
    assert enc.solve(Instance(1, chunks=2)) == None
    assert enc.solve(Instance(2, chunks=2)) != None

def test_dgx1_noncombining():
    topo = dgx1()
    enc = PathEncoding(topo, allgather(topo.num_nodes()))
    assert enc.solve(Instance(1)) == None
    assert enc.solve(Instance(2)) != None

def test_dgx1_combining_reducible():
    topo = dgx1()
    enc = PathEncoding(topo, reduce_scatter(topo.num_nodes()))
    assert enc.solve(Instance(1)) == None
    assert enc.solve(Instance(2)) != None

def test_dgx1_combining_nonreducible():
    topo = dgx1()
    enc = PathEncoding(topo, allreduce(topo.num_nodes()))
    assert enc.solve(Instance(1)) == None
    assert enc.solve(Instance(2)) != None

def test_memory_constraint():
    topo = line(3)
    enc = PathEncoding(topo, alltoall(topo.num_nodes()))
    assert enc.solve(Instance(2, extra_memory=0)) == None
    assert enc.solve(Instance(2, extra_memory=1)) != None
