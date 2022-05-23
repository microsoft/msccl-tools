# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.collectives import *

def null_collective(num_nodes):
    return build_collective(f'Null(n={num_nodes})', num_nodes, 1,
        lambda r, c: True, lambda r, c: False)

def impossible_collective(num_nodes):
    return build_collective(f'Impossible(n={num_nodes})', num_nodes, 1,
        lambda r, c: False, lambda r, c: True)
