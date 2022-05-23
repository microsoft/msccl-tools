# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .common import *
from msccl.serialization import MSCCLEncoder, MSCCLDecoder
from msccl.algorithm import Algorithm, Step
from msccl.topologies import fully_connected
from msccl.instance import Instance

def test_algorithm_roundtrip():
    name = 'test_algorithm'
    num_nodes = 2
    collective = null_collective(num_nodes)
    topo = fully_connected(num_nodes)
    steps = [Step(1,[(0,0,1)]),Step(2,[(1,1,0),(1,0,1)]),Step(1,[(0,1,0)])]
    instance = Instance(3, pipeline=2)
    algo1 = Algorithm(name, collective, topo, instance, steps)
    json = MSCCLEncoder().encode(algo1)
    assert json != None

    algo2 = MSCCLDecoder().decode(json)
    assert algo2.name == name
    assert algo2.instance == instance
    assert algo2.steps == steps
