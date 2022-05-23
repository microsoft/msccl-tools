# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.ncd_reduction import non_combining_dual
from msccl.topologies import reverse_topology
from z3 import *
from fractions import Fraction

def _flow(chunk, src, dst):
    return Real(f'flow_{chunk}_from_{src}_to_{dst}')

def lower_bound_rounds(topology, collective, logging=False):
    '''
    Solve a lower bound rounds required by any algorithm. Uses a multi-commodity feasibility inspired encoding to Z3.
    '''

    opt = Optimize()

    # Remember names before possible non-combining dual reduction
    collective_name = collective.name
    topology_name = topology.name

    # Use non-combining dual if necessary
    if collective.is_combining:
        collective = non_combining_dual(collective)
        topology = reverse_topology(topology)

    chunks = collective.chunks()
    ranks = collective.ranks()

    for chunk in chunks:
        for rank in ranks:
            # All flows are between 0 and 1
            for dst in topology.destinations(rank):
                opt.add(_flow(chunk,rank,dst) >= 0)
                opt.add(_flow(chunk,rank,dst) <= 1)
            total_in = sum(_flow(chunk,src,rank) for src in topology.sources(rank))
            if not collective.precondition(rank, chunk):
                # Ranks not in the precondition need to justify outflows
                for dst in topology.destinations(rank):
                    opt.add(_flow(chunk,rank,dst) <= total_in)
                # Ranks in the postcondition, but not in the precondition need the whole chunk
                if collective.postcondition(rank, chunk):
                    opt.add(total_in == 1)

    # Represents how many rounds all the steps of the algorithm would use
    rounds = Real(f'rounds')
    
    for srcs, dsts, bw, _ in topology.bandwidth_constraints():
        # Sum of all flows relevant to this constraint
        sum_flow = sum(_flow(chunk,src,dst) for src in srcs for dst in dsts for chunk in chunks)
        # Total flow must be less than the limit, taking rounds into consideration
        opt.add(sum_flow <= bw * rounds)

    # Minimize the number of rounds
    min_rounds = opt.minimize(rounds)
    result = opt.check()
    if result == sat:
        bound_ref = opt.lower(min_rounds)
        if isinstance(bound_ref, IntNumRef):
            rounds_lb = Fraction(bound_ref.as_long(), 1)
        elif isinstance(bound_ref, RatNumRef):
            rounds_lb = bound_ref.as_fraction()
        else:
            raise RuntimeError(f'Unhandled Z3 numeral type: {type(bound_ref)}')
        if logging:
            print(f'{collective_name} algorithms need at least {rounds_lb} rounds in {topology_name} topology.')
        return rounds_lb
    else:
        if logging:
            if result == unsat:
                print(f'Unsat. {collective_name} is not implementable in {topology_name} topology.')
            else:
                assert result == unknown, 'Unhandled Z3 result'
                print('Unknown. Z3 was not able to solve the lower bound.')
        return None
