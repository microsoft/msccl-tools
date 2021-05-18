# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
from sccl.topologies import Topology
from sccl.collectives import build_collective
from sccl.rounds_bound import *

def test_rounds_bound_unimplementable():
    topo = Topology('Unconnected', [[0,0],[0,0]])
    coll = build_collective('Send', 2, 1, lambda r, c: r == 0, lambda r, c: r == 1)
    assert lower_bound_rounds(topo, coll) == None
