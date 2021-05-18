# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.permutation import Permutation
from z3 import *

def _pn(node):
    return Int(f'perm_node_{node}')

def _pc(chunk):
    return Int(f'perm_chunk_{chunk}')

def _select_node_permutation(s, topology):
    # Select a permutation of nodes
    for node in topology.nodes():
        s.add(_pn(node) >= 0)
        s.add(_pn(node) < topology.num_nodes())
        for prev in range(node):
            s.add(_pn(node) != _pn(prev))

def _select_chunk_permutation(s, collective):
    # Select a permutation of chunks
    for chunk in collective.chunks():
        s.add(_pc(chunk) >= 0)
        s.add(_pc(chunk) < collective.num_chunks)
        for prev in range(chunk):
            s.add(_pc(chunk) != _pc(prev))

def _decode_permutation(model, topology, collective = None):
    node_permutation = [model.eval(_pn(node)).as_long() for node in topology.nodes()]
    if collective != None:
        chunk_permutation = [model.eval(_pc(chunk)).as_long() for chunk in collective.chunks()]
    else:
        chunk_permutation = None
    return Permutation(node_permutation, chunk_permutation)

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

def find_isomorphisms(topology, target_topology, limit=None, logging=False):
    '''
    Finds all isomorphisms from one topology to a target topology. Returns a list of permutations.
    '''
    if len(topology.switches) > 0:
        raise ValueError('Topologies with switches are not supported.')

    if limit != None and limit <= 0:
        return []
    
    if topology.num_nodes() != target_topology.num_nodes():
        return []

    if logging:
        print(f'Finding all isomorphisms from {topology.name} to {target_topology.name}')

    s = Solver()

    _select_node_permutation(s, topology)
    s.add(_links_constraint(topology, target_topology))

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

def _conditions_constraint(collective):
    def conditions_isomorphic(perm_rank, perm_chunk, pre, post):
        for rank in collective.ranks():
            for chunk in collective.chunks():
                if collective.precondition(rank, chunk) != pre:
                    yield Not(And(perm_rank == rank, perm_chunk == chunk))
                if collective.postcondition(rank, chunk) != post:
                    yield Not(And(perm_rank == rank, perm_chunk == chunk))
    conditions = []
    for rank in collective.ranks():
        for chunk in collective.chunks():
            pre = collective.precondition(rank, chunk)
            post = collective.postcondition(rank, chunk)
            conditions.extend(conditions_isomorphic(_pn(rank), _pc(chunk), pre, post))
    return And(conditions)

def _addresses_constraint(collective):
    if not collective.is_combining:
        return True

    def addresses_isomorphic(perm_chunk1, perm_chunk2, same):
        for chunk1 in collective.chunks():
            for chunk2 in collective.chunks():
                if (collective.address(chunk1) == collective.address(chunk2)) != same:
                    yield Not(And(perm_chunk1 == chunk1, perm_chunk2 == chunk2))
    conditions = []
    for chunk1 in collective.chunks():
        for chunk2 in collective.chunks():
            same = collective.address(chunk1) == collective.address(chunk2)
            conditions.extend(addresses_isomorphic(_pc(chunk1), _pc(chunk2), same))
    return And(conditions)

def find_automorphisms(topology, collective=None, exclude_trivial=True, logging=False):
    '''
    Finds all automorphisms of a topology. Returns a list of permutations (each being a list of node numbers) or None
    if all permutations are automorphisms of the topology.
    '''
    if len(topology.switches) > 0:
        raise ValueError('Topologies with switches are not supported.')

    if logging:
        if collective != None:
            print(f'Finding all automorphisms for {topology.name} with {collective.name}.')
        else:
            print(f'Finding all automorphisms for {topology.name}.')

    s = Solver()

    _select_node_permutation(s, topology)
    s.add(_links_constraint(topology, topology))

    if collective != None:
        _select_chunk_permutation(s, collective)
        s.add(_conditions_constraint(collective))
        s.add(_addresses_constraint(collective))

    if exclude_trivial:
        assignment = [_pn(node) == node for node in topology.nodes()]
        if collective != None:
            assignment.extend([_pc(chunk) == chunk for chunk in collective.chunks()])
        s.add(Not(And(assignment)))

    automorphisms = []
    while s.check() == sat:
        automorphism = _decode_permutation(s.model(), topology, collective)
        automorphisms.append(automorphism)

        if logging:
            print(automorphism)

        # Block this permutation
        assignment = [_pn(node) == perm for node, perm in enumerate(automorphism.nodes)]
        if collective != None:
            assignment.extend([_pc(chunk) == perm for chunk, perm in enumerate(automorphism.chunks)])
        s.add(Not(And(assignment)))

    if logging:
        print(f'{len(automorphisms)} automorphisms found.')
    return automorphisms

def are_all_permutations_automorphisms(topology, collective=None, logging=False):
    if len(topology.switches) > 0:
        raise ValueError('Topologies with switches are not supported.')

    if logging:
        if collective != None:
            print(f'Checking if all permutations are automorphisms for {topology.name} with {collective.name}.')
        else:
            print(f'Checking if all permutations are automorphisms for {topology.name}.')

    s = Solver()

    _select_node_permutation(s, topology)
    not_isomorphic = []
    not_isomorphic.append(Not(_links_constraint(topology, topology)))

    if collective != None:
        _select_chunk_permutation(s, collective)
        not_isomorphic.append(Not(_conditions_constraint(collective)))
        not_isomorphic.append(Not(_addresses_constraint(collective)))

    s.add(Or(not_isomorphic))

    # Unsat means that there is no non-automorphic counterexample
    result = s.check()
    if result == sat:
        if logging:
            print(f'Not all permutations are automorphisms. Counterexample: {_decode_permutation(s.model(), topology, collective)}')
        return False
    else:
        assert result == unsat
        print(f'All permutations are automorphisms.')
        return True
