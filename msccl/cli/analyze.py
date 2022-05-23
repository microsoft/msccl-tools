# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .known_topologies import KnownTopologies
from .known_collectives import KnownCollectives
from .common import *
from msccl.rounds_bound import lower_bound_rounds
from msccl.isomorphisms import find_isomorphisms

def make_analyses(cmd_parsers):
    handler_funcs = []
    handler_funcs.append(make_handle_bound_rounds)
    handler_funcs.append(make_handle_find_isomorphisms)

    return make_cmd_category(cmd_parsers, 'analyze', 'analysis', handler_funcs)

def make_handle_bound_rounds(cmd_parsers):
    cmd = cmd_parsers.add_parser('rounds')
    topologies = KnownTopologies(cmd)
    collectives = KnownCollectives(cmd)

    def handle(args, command):
        if command != 'rounds':
            return False

        topology = topologies.create(args)
        collective = collectives.create(args, topology.num_nodes())
        lower_bound_rounds(topology, collective, logging=True)
        return True
    
    return handle

def make_handle_find_isomorphisms(cmd_parsers):
    cmd = cmd_parsers.add_parser('isomorphisms')
    topologies1 = KnownTopologies(cmd, tag='1')
    topologies2 = KnownTopologies(cmd, tag='2')

    def handle(args, command):
        if command != 'isomorphisms':
            return False

        topology1 = topologies1.create(args)
        topology2 = topologies2.create(args)
        isomorphisms = find_isomorphisms(topology1, topology2, logging=True)
        return True
    
    return handle
