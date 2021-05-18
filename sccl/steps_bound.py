# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

def _distances(topology):
    # Floyd–Warshall algorithm for all-pairs shortest paths
    nodes = range(topology.num_nodes())
    dist = [[math.inf for _ in nodes] for _ in nodes]
    for dst in nodes:
        for src in topology.sources(dst):
            dist[src][dst] = 1
    for node in nodes:
        dist[node][node] = 0
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

def lower_bound_steps(topology, collective):
    ''' Finds a lower bound for the steps required as the maximum distance for a chunk from any of its sources. '''

    dist = _distances(topology)

    # Find the maximum of the least steps required for each chunk 
    least_steps = 0
    for chunk in collective.chunks():
        for dst in collective.ranks():
            if collective.postcondition(dst, chunk):
                # Find the shortest distance from some rank in the precondition
                least_distance = math.inf
                for src in collective.ranks():
                    if collective.precondition(src, chunk):
                        least_distance = min(least_distance, dist[src][dst])
                # Update the least steps required if the distance from any rank in the precondition is larger
                least_steps = max(least_steps, least_distance)

    if least_steps == math.inf:
        # Return None if the collective is unimplementable with any number of steps
        return None
    else:
        return least_steps
