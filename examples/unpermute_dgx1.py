# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script shows how to use SCCL to find a way to permute the nodes of a DGX1 to match the default order.

from sccl.topologies import *
from sccl.isomorphisms import find_isomorphisms

def solve_dgx1_permutation():
    local = nvlink_only()
    isomorphisms = find_isomorphisms(dgx1(), local, limit=1)
    if len(isomorphisms) == 0:
        raise RuntimeError('No isomorphism to DGX1 found')
    return isomorphisms[0].nodes
