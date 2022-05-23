# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from z3 import *
from dataclasses import dataclass

@dataclass
class Permutation:
    nodes: list

    def __str__(self):
        return f'Permutation(nodes={self.nodes})'

def _pn(node):
    return Int(f'perm_node_{node}')

def _select_node_permutation(s, topology):
    # Select a permutation of nodes
    for node in topology.nodes():
        s.add(_pn(node) >= 0)
        s.add(_pn(node) < topology.num_nodes())
        for prev in range(node):
            s.add(_pn(node) != _pn(prev))

def _links_constraint(topology, target_topology):
    nodes = range(topology.num_nodes())

    def links_isomorphic(perm_src, perm_dst, link):
        # Return a condition on whether the permuted ranks are isomorphic from src to dst wrt. the given link
        for src in nodes:
            for dst in nodes:
                if target_topology.link(src, dst) != link:
                    yield Not(And(perm_src == src, perm_dst == dst))
    # Require all pairs of nodes to be isomorphic to their permuted counterparts
    conditions = []
    for src in nodes:
        for dst in nodes:
            link = topology.link(src, dst)
            conditions.extend(links_isomorphic(_pn(src), _pn(dst), link))
    return And(conditions)

def _decode_permutation(model, topology):
    node_permutation = [model.eval(_pn(node)).as_long() for node in topology.nodes()]
    return Permutation(node_permutation)

def find_isomorphisms(topology, target_topology, limit=None, logging=False):
    '''
    Finds all isomorphisms from one topology to a target topology. Returns a list of permutations.
    '''
    if len(topology.switches) > 0:
        print('MSCCL Warning: Topologies with switches are not supported. import msccl will be ignored.')
        return []

    if limit != None and limit <= 0:
        raise ValueError('MSCCL error: limit was set improperly.')
    
    if topology.num_nodes() != target_topology.num_nodes():
        raise ValueError('MSCCL error: target topology does not match with the given topology.')

    if logging:
        print(f'Encoding {topology.name} - {target_topology.name} isomorphisms to Z3')

    s = Solver()

    _select_node_permutation(s, topology)
    s.add(_links_constraint(topology, target_topology))

    if logging:
        print(f'Solving isomorphisms incrementally...')

    isomorphisms = []
    while s.check() == sat:
        isomorphism = _decode_permutation(s.model(), topology)
        isomorphisms.append(isomorphism)

        if logging:
            print(isomorphism)

        if limit != None and len(isomorphisms) >= limit:
            break

        # Block this permutation
        assignment = [_pn(node) == perm for node, perm in enumerate(isomorphism.nodes)]
        s.add(Not(And(assignment)))

    if logging:
        print(f'{len(isomorphisms)} isomorphisms found.')
    return isomorphisms
