# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.collectives import alltoall
from msccl.algorithm import *
from msccl.instance import *

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import math

@dataclass
class _BwLimit:
    srcs: set
    dsts: set
    bw: int
    util: int = 0

def synthesize_greedy_distributed_alltoall(topology, local_algorithm, logging=False):
    if local_algorithm.is_pipelined():
        raise ValueError('Pipelining is not supported.')

    chunks = local_algorithm.instance.chunks
    local_nodes = local_algorithm.topology.num_nodes()
    local_alltoall = alltoall(local_nodes).chunk_up(chunks)
    try:
        local_algorithm.check_implements(local_alltoall)
    except:
        raise ValueError(f'Given local Alltoall algorithm "{local_algorithm.name}" does not implement Alltoall in the local topology.')

    if topology.num_nodes() % local_nodes != 0:
        raise ValueError(f'Number of nodes in topology is not a multiple of ranks in local_algorithm.')
    num_copies = topology.num_nodes() // local_nodes

    def is_pair_remote(rank1, rank2):
        # A pair of nodes is remote if they are in different copies of the local topology
        return src // local_nodes != dst // local_nodes

    # Check that all switches are either purely local or remote.
    # Also check that remote part is fully connected.
    # Also remember remote constraints.
    remote_constraints = []
    for srcs, dsts, bw, _ in topology.bandwidth_constraints():
        has_local_pairs = False
        has_remote_pairs = False
        for src in srcs:
            for dst in dsts:
                if is_pair_remote(src, dst):
                    has_remote_pairs = True
                else:
                    has_local_pairs = True
        if has_local_pairs and has_remote_pairs:
            raise NotImplementedError('Support for switches with mixed local and remote connections is not implemented.')
        if has_remote_pairs and bw == 0:
            # This is required because it's what makes Alltoall routing trivial
            raise ValueError('All remote pairs must have direct connectivity.')
        if has_remote_pairs:
            remote_constraints.append((srcs, dsts, bw))

    collective = alltoall(topology.num_nodes())

    def nth_chunk_for_pair(src, dst, idx):
        # The following chunk calculation respects both the _scattered and _transpose
        # pre/postconditions in Alltoall. When substituting it in:
        # -the precondition (chunk % self.num_nodes) simplifies to src
        # -the postcondition ((chunk // self.num_nodes) % self.num_nodes) simplifies to dst
        return (src + dst * collective.num_nodes) * chunks + idx

    if logging:
        print('Generating sends for remote pairs')

    # Generate all of the sends that need to happen for the remote part, grouped by pairs of src and dst
    remote_sends = {}
    for src in collective.ranks():
        for dst in collective.ranks():
            if is_pair_remote(src, dst):
                sends = [(nth_chunk_for_pair(src, dst, i), src, dst)
                    for i in reversed(range(chunks))]
                remote_sends[(src,dst)] = sends

    # This function pulls as many sends out of remote_sends as the topology's bw constraints allow
    def pack_sends(rounds):
        packed_sends = []
        # Make a mutable copy of the bandwidth constraints
        bw_limits = [_BwLimit(srcs, dsts, bw * rounds) for srcs, dsts, bw in remote_constraints]
        empty_pairs = []
        for pair in remote_sends:
            src, dst = pair
            sends = remote_sends[pair]
            # Yield as many sends as allowed by the bw limits
            max_sends = len(sends)
            relevant_limits = []
            for limit in bw_limits:
                if src in limit.srcs and dst in limit.dsts:
                    max_sends = min(max_sends, limit.bw - limit.util)
                    relevant_limits.append(limit)
            for i in range(max_sends):
                packed_sends.append(sends.pop())
            # Remove used bandwidth from limits
            for limit in relevant_limits:
                limit.util += max_sends
            if len(sends) == 0:
                empty_pairs.append(pair)
        # Remove pairs that don't have sends remaining
        for pair in empty_pairs:
            del remote_sends[pair]
        return packed_sends

    steps = []

    if logging:
        print('Overlapping remote sends with local algorithm')

    for step_idx, local_step in enumerate(local_algorithm.steps):
        sends = []

        # Translate copies of the local algorithm to the new space of ranks
        for chunk, src, dst in local_step.sends:
            for i in range(num_copies):
                # Translates ranks from the local to the distributed topology
                def to_dist(rank):
                    return rank + i * local_nodes

                # Calculate origin and target ranks that match the Alltoall pre/postconditions
                origin = (chunk // chunks) % local_nodes
                target = (chunk // chunks) // local_nodes

                # Check that we got that calculation right
                assert local_alltoall.precondition(origin, chunk)
                assert local_alltoall.postcondition(target, chunk)

                # Get the chunk number in the distributed algorithm
                chunk_idx = chunk % chunks
                dist_chunk = nth_chunk_for_pair(to_dist(origin), to_dist(target), chunk_idx)

                # Translate send src and dst to distributed space and the send to the distributed algorithm
                sends.append((dist_chunk, to_dist(src), to_dist(dst)))

        # Pack sends respecting the local step's rounds
        packed_sends = pack_sends(local_step.rounds)
        sends.extend(packed_sends)
        if logging:
            print(f'Packed {len(packed_sends)} remote sends into step {step_idx+1}')

        steps.append(Step(local_step.rounds, sends))
    
    # If any remote sends are left over once the local algorithm is done, put them all into the last step
    remaining_sends = len(remote_sends)
    if remaining_sends > 0:
        last_step = steps[-1]

        # Add sends and count their utilization against all constraints
        bw_limits = [_BwLimit(srcs, dsts, bw) for srcs, dsts, bw in remote_constraints]
        empty_pairs = []
        for pair in remote_sends:
            src, dst = pair
            sends = remote_sends[pair]
            # Add utilization against all relevant limits
            for limit in bw_limits:
                if src in limit.srcs and dst in limit.dsts:
                    limit.util += len(sends)
            # Add the sends to the last step
            last_step.sends.extend(sends)

        # Find the least rounds required and add additional rounds to last step
        additional_rounds = max(math.ceil(limit.util / limit.bw) for limit in bw_limits)
        last_step.rounds += additional_rounds
        if logging:
            print(f'Packed remaining {remaining_sends} remote sends into step {len(steps)} by adding {additional_rounds} additional rounds')
    else:
        if logging:
            print('All remote sends fit into the rounds of the local algorithm')
        additional_rounds = 0

    instance = local_algorithm.instance.set(extra_rounds=local_algorithm.instance.extra_rounds + additional_rounds)
    return Algorithm.make_implementation(collective, topology, instance, steps)
