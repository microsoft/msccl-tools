# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
from .common import *
from msccl.algorithm import Algorithm, Step
from msccl.topologies import fully_connected
from msccl.instance import Instance

def test_invalid_empty():
    with pytest.raises(RuntimeError):
        num_nodes = 2
        topo = fully_connected(num_nodes)
        algo = Algorithm.make_implementation(impossible_collective(num_nodes), topo, Instance(1), [Step(1,[])])

def test_valid_empty():
    num_nodes = 2
    topo = fully_connected(num_nodes)
    algo = Algorithm.make_implementation(null_collective(num_nodes), topo, Instance(1), [Step(1,[])])
    assert algo != None
