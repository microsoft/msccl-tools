# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from collections import defaultdict

import math

def _distances(topology):
    # Floyd–Warshall algorithm for all-pairs shortest paths with path information
    # Modified to track all shortest paths
    nodes = range(topology.num_nodes())
    dist = [[math.inf for _ in nodes] for _ in nodes]
    next = [[set() for _ in nodes] for _ in nodes]
    for dst in nodes:
        for src in topology.sources(dst):
            dist[src][dst] = 1
            next[src][dst].add(dst)
    for node in nodes:
        dist[node][node] = 0
        next[node][node].add(node)
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][j] >= dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next[i][j].update(next[i][k])
    return dist, next

def shortest_path_sets(topology, collective):
    dist, next = _distances(topology)

    spsets = {}
    for id, chunk in enumerate(collective._chunks):
        spset = set()
        for u in chunk.precondition:
            for v in chunk.postcondition:
                curr = next[u][v]
                if not curr:
                    continue
                spset.add(u)
                while not v in curr:
                    curr = set().union(*[next[x][v] for x in curr])
                    spset.update(curr)
                spset.update(curr)
        spsets[id] = spset

    return spsets
