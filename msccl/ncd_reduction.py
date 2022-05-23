# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.collectives import *
from msccl.topologies import reverse_topology
from msccl.algorithm import Algorithm, Step
from collections import defaultdict

class ReductionNotApplicableError(ValueError):
    pass

def non_combining_dual(primal):
    if not primal.is_combining:
        raise ReductionNotApplicableError('The collective is already non-combining.')

    if primal.has_triggers():
        raise ReductionNotApplicableError('The collective has triggers.')

    dual_precondition = defaultdict(set)
    dual_postcondition = defaultdict(set)

    addresses = set()
    for chunk in primal.chunks():
        addr = primal.address(chunk)
        addresses.add(addr)
        for rank in primal.ranks():
            if primal.postcondition(rank, chunk):
                dual_precondition[addr].add(rank)
            if primal.precondition(rank, chunk):
                dual_postcondition[addr].add(rank)
    for addr in dual_precondition:
        if len(dual_precondition[addr]) > 1:
            raise ReductionNotApplicableError('The non-combining reduction is only applicable to collectives with a unique root per address.')

    return build_collective(f'Dual{primal.name}', primal.num_nodes, len(addresses),
        lambda r, c: r in dual_precondition[c],
        lambda r, c: r in dual_postcondition[c])

def recover_primal_algorithm(dual_algorithm, primal, original_topology, instance):
    primal_steps = []
    for step in reversed(dual_algorithm.steps):
        primal_sends = [(chunk, dst, src) for chunk, src, dst in step.sends]
        primal_steps.append(Step(step.rounds, primal_sends))
    return Algorithm.make_implementation(primal, original_topology, instance, primal_steps)

def wrap_try_ncd_reduction(solver_cls):
    class NonCombiningReductionWrapper(solver_cls):
        def __init__(self, topology, collective):
            self.primal = collective
            try:
                # Create the dual collective
                self.dual = non_combining_dual(collective)
                collective = self.dual

                # Solve the dual in the reverse topology
                self.original_topology = topology
                topology = reverse_topology(topology)
            except ReductionNotApplicableError:
                self.dual = None
            super().__init__(topology, collective)

        def solve(self, instance):
            algo = super().solve(instance)
            if self.dual != None and algo != None:
                return recover_primal_algorithm(algo, self.primal, self.original_topology, instance)
            else:
                return algo

    return NonCombiningReductionWrapper
