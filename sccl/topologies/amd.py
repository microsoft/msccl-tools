# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .topology import Topology

def amd4():
    links = [
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ]
    return Topology('AMD4', links)

def amd8():
    links = [
        [0, 5, 6, 6, 5, 6, 5, 5],
        [5, 0, 5, 5, 6, 5, 6, 6],
        [6, 5, 0, 6, 5, 6, 5, 5],
        [6, 5, 6, 0, 5, 6, 5, 5],
        [5, 6, 5, 5, 0, 5, 6, 6],
        [6, 5, 6, 6, 5, 0, 5, 5],
        [5, 6, 5, 5, 6, 5, 0, 6],
        [5, 6, 5, 5, 6, 5, 6, 0]
    ]
    return Topology('AMD8', links)
