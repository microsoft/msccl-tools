# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Step(object):
    rounds: int
    sends: list

class Algorithm(object):
    def __init__(self, name, collective, topology, instance, steps, input_map = {}, output_map = {}):
        self.name = name
        self.topology = topology
        self.collective = collective
        self.instance = instance
        self.steps = steps
        self.input_map = input_map
        self.output_map = output_map

        self._update_link_utilizations()
        self._check_bandwidth_constraints()

        for step in self.steps:
            step.sends.sort()

    @classmethod
    def make_implementation(cls, collective, topology, instance, steps):
        chunked = collective.chunk_up(instance.chunks)

        # Figure out input and output addresses
        input_map = {}
        output_map = {}
        for rank in chunked.ranks():
            input_addrs = set()
            output_addrs = set()
            for chunk in chunked.chunks():
                # An address is an input address if any of its chunks is in the precondition
                if chunked.precondition(rank, chunk):
                    input_addrs.add(chunked.address(chunk))
                # An address is an output address if any of its chunks is in the postcondition
                if chunked.postcondition(rank, chunk):
                    output_addrs.add(chunked.address(chunk))
            if len(input_addrs) > 0:
                input_map[rank] = input_addrs
            if len(output_addrs) > 0:
                output_map[rank] = output_addrs

        # Concatenate collective and topology names plus instance arguments to create a name
        name = f'{collective.name}-{topology.name}-{instance}'

        algo = cls(name, collective, topology, instance, steps, input_map, output_map)
        algo.check_implements(chunked)
        if instance.extra_rounds > 0:
            used_extra_rounds = algo.extra_rounds()
            if used_extra_rounds > instance.extra_rounds:
                raise ValueError(f'steps use {used_extra_rounds} extra rounds but only {instance.extra_rounds} were allowed')
        return algo

    def ranks(self):
        return range(self.topology.num_nodes())
    
    def num_steps(self):
        return len(self.steps)

    def extra_rounds(self):
        rounds = 0
        for step in self.steps:
            rounds += step.rounds
        return rounds - self.num_steps()

    def is_pipelined(self):
        return self.instance.pipeline != None

    def check_implements(self, collective):
        if self.topology.num_nodes() != collective.num_nodes:
            raise RuntimeError('topology and collective have different number of nodes')
        # Find which chunks will be sent from an address
        chunks_at_address = defaultdict(list)
        for chunk in collective.chunks():
            chunks_at_address[collective.address(chunk)].append(chunk)
        # State records if a rank holds a chunk
        def idx(rank, chunk):
            return rank * collective.num_chunks + chunk
        state = [False] * (collective.num_nodes * collective.num_chunks)
        # Initialize state from precondition
        for rank in collective.ranks():
            for chunk in collective.chunks():
                state[idx(rank, chunk)] = collective.precondition(rank, chunk)
        # Propagate state through sends of every step
        for step in self.steps:
            next_state = state.copy()
            for addr, src, dst in step.sends:
                for chunk in chunks_at_address[addr]:
                    next_state[idx(dst, chunk)] |= state[idx(src, chunk)]
            state = next_state
        # Check that the postcondition holds
        for rank in collective.ranks():
            for chunk in collective.chunks():
                if collective.postcondition(rank, chunk) and not state[idx(rank, chunk)]:
                    raise RuntimeError(f'rank {rank} does not get chunk {chunk} as required by the postcondition')

    def _update_link_utilizations(self):
        self._link_utilizations = []
        ranks = range(self.topology.num_nodes())
        for step in self.steps:
            step_utilizations = [[0 for _ in ranks] for _ in ranks]
            for addr, src, dst in step.sends:
                step_utilizations[dst][src] += 1 # Same order as topology
            self._link_utilizations.append(step_utilizations)

    def _check_bandwidth_constraints(self):
        for srcs, dsts, bw, name in self.topology.bandwidth_constraints():
            for step_num, step in enumerate(self.steps):
                util = 0
                for dst in dsts:
                    for src in srcs:
                        if self.is_pipelined():
                            for overlapping_step in range(step_num, len(self.steps), self.instance.pipeline):
                                util += self._link_utilizations[overlapping_step][dst][src]
                        else:
                            util += self._link_utilizations[step_num][dst][src]
                assert util <= bw * step.rounds, \
                    f'Step {step_num} uses {util} bandwidth but constraint {name} only allows for {bw * step.rounds} bandwidth (when rounds={step.rounds}).'

    def __str__(self):
        s = ''
        for i, step in enumerate(self.steps):
            if i != 0:
                s += '\n'
            if step.rounds > 1:
                s += f'(step {i+1}, rounds={step.rounds}) '
            else:
                s += f'(step {i+1}) '
            s += ', '.join([f'{chunk}:{src}→{dst}' for chunk, src, dst in step.sends])
        return s
