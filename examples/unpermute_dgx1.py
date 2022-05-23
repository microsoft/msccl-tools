# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script shows how to use MSCCL to find a way to permute the nodes of a DGX1 to match the default order.

from msccl.topologies import *
from msccl.isomorphisms import find_isomorphisms

def solve_dgx1_permutation():
    local = nvlink_only()
    isomorphisms = find_isomorphisms(dgx1(), local, limit=4)
    if len(isomorphisms) == 0:
        raise RuntimeError('No isomorphism to DGX1 found')
    return isomorphisms
print(solve_dgx1_permutation())
