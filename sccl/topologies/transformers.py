# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .topology import Topology

def reverse_topology(topology):
    '''
    Reverses the direction of all links and switches in the topology.
    '''
    num_nodes = topology.num_nodes()
    # Transpose the links
    links = [[topology.links[src][dst] for src in range(num_nodes)] for dst in range(num_nodes)]
    # Reverse the switches
    switches = [(dsts, srcs, bw, f'{name}_reversed') for srcs, dsts, bw, name in topology.switches]
    return Topology(f'Reverse{topology.name}', links, switches)

def binarize_topology(topology):
    '''
    Makes all link bandwidths 1 and removes all switches. Essentially, the bandwidth modeling part of the topology
    is stripped out and only connectivity information is kept.
    '''
    num_nodes = topology.num_nodes()
    links = [[1 if topology.links[src][dst] > 0 else 0 for src in range(num_nodes)] for dst in range(num_nodes)]
    return Topology(f'Binarized{topology.name}', links, [])
