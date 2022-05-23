# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.algorithm import *
from msccl.ncd_reduction import wrap_try_ncd_reduction
from z3 import *

from collections import defaultdict

def _start(chunk, rank):
    return Int(f'start_{chunk}_at_{rank}')

def _end(chunk, rank):
    return Int(f'end_{chunk}_at_{rank}')

def _rounds(step):
    return Int(f'rounds_{step}')

def _send(chunk, src, dst):
    return Bool(f'send_{chunk}_from_{src}_to_{dst}')

def _sent_in(chunk, src, dst, step):
    # Constructs a Z3 term that is true iff a chunk is sent from src to dst in step
    return And(_send(chunk, src, dst), _start(chunk, dst) == step + 1)

def _idx(addr, rank):
    return Int(f'idx_{addr}_at_{rank}')

def _addr_start(addr, rank):
    return Int(f'addr_start_{addr}_at_{rank}')

def _addr_end(addr, rank):
    return Int(f'addr_end_{addr}_at_{rank}')

class PathEncodingBase(object):
    def __init__(self, topology, collective):
        self.topology = topology
        self.collective = collective

    def _encode(self, s, instance, collective):
        # Calculate how much iterations of the algorithm overlap if pipelining is specified
        if instance.pipeline != None:
            # TODO: move this check into Instance
            if instance.pipeline <= 0:
                raise ValueError('instance.pipeline must be strictly positive.')
            overlap = max(instance.steps - instance.pipeline, 0)
        else:
            overlap = 0

        # Correctness
        for chunk in collective.chunks():
            for rank in collective.ranks():
                if collective.precondition(rank, chunk):
                    # Have chunks start on their starting ranks before the first step
                    # This is not required for the encoding, but makes debugging the models produced more intuitive
                    s.add(_start(chunk, rank) == 0)
                else:
                    # Any rank that gets a chunk (and doesn't start with it) must have a unique source for it
                    sent_once = PbEq([(_send(chunk, src, rank), 1) for src in self.topology.sources(rank)], 1)
                    s.add(Implies(_start(chunk, rank) <= instance.steps, sent_once))
                # If the postcondition requires the chunk on the rank then it must start being there before the end
                if collective.postcondition(rank, chunk):
                    s.add(_start(chunk, rank) <= instance.steps)
                for src in self.topology.sources(rank):
                    # If a rank send a chunk then it needs to have it before sending it
                    s.add(Implies(_send(chunk, src, rank), _start(chunk, src) < _start(chunk, rank)))
                    if instance.extra_memory != None:
                        # Also to send a chunk it needs to not have been deleted before sending it
                        s.add(Implies(_send(chunk, src, rank), _end(chunk, src) >= _start(chunk, rank) - 1))
                    # Handle chunks at the same address getting reduced in combining collectives
                    if collective.is_combining:
                        for other in collective.chunks():
                            if other != chunk and collective.address(other) == collective.address(chunk):
                                # If you send and another chunk at the same address is available (i.e. reduced) then you have to send that too at the same time
                                s.add(Implies(And(_send(chunk, src, rank), _start(other, src) < _start(chunk, rank)),
                                                And(_send(other, src, rank), (_start(other, rank) == _start(chunk, rank)))))

                    # Handle the triggers used in subproblem based synthesizers
                    if collective.trigger(rank, chunk) != None:
                        # When receiving a chunk with a trigger, the triggering chunk must be sent at the same time
                        trigger = collective.trigger(rank, chunk)
                        s.add(Implies(_send(chunk, src, rank),
                            And(_send(trigger, rank, src), _start(trigger, src) == _start(chunk, rank))))
                    if collective.trigger(src, chunk) != None:
                        # When sending a chunk with a trigger, the triggering chunk must be received at the same time
                        trigger = collective.trigger(src, chunk)
                        s.add(Implies(_send(chunk, src, rank),
                            And(_send(trigger, rank, src), _start(trigger, src) == _start(chunk, rank))))

        # Rounds
        # Each step must use at least one round of bandwidth
        s.add(*[_rounds(step) >= 1 for step in range(instance.steps)])
        # Total number of rounds used by all steps must not exceed the limits given
        s.add(sum([_rounds(step) for step in range(instance.steps)]) <= instance.rounds())
        # Overlapping steps in pipelined algorithms must use the same number of rounds
        for step in range(instance.steps - overlap):
            for overlapping_step in range(step, instance.steps, instance.steps - overlap):
                if overlapping_step != step:
                    s.add(_rounds(step) == _rounds(overlapping_step))

        # Bandwidth
        # Each bandwidth group (e.g. a link or a switch) generates a separate set of constraints
        for srcs, dsts, bw, _ in self.topology.bandwidth_constraints():
            # overlap is subtracted here because overlapping steps are considered together
            for step in range(instance.steps - overlap):
                pb_sends = []
                for src in srcs:
                    for dst in dsts:
                        # Generate terms for all sends on this step and group them by address and edge
                        sends_by_addr = defaultdict(list)
                        for chunk in collective.chunks():
                            # Consider all pipelined steps that overlap with this step
                            for overlapping_step in range(step, instance.steps, instance.steps - overlap):
                                sends_by_addr[(collective.address(chunk))].append(_sent_in(chunk, src, dst, overlapping_step))
                        # Count sends happening on an address only once and give each of these weight 1
                        pb_sends.extend([(Or(sends),1) for sends in sends_by_addr.values()])
                # For each number of rounds this step could have impose a pseudo-boolean
                # constraint limiting sends on this step to the available bandwidth
                for i in range(1, instance.extra_rounds + 2):
                    s.add(Implies(_rounds(step) == i, PbLe(pb_sends, bw * i)))

        # Memory
        if instance.extra_memory != None:
            # Choose the last step a chunk is present on a rank
            for chunk in collective.chunks():
                for rank in collective.ranks():
                    if collective.postcondition(rank, chunk):
                        # In the postcondition the chunk can not stop being on the rank before the end of the algorithm
                        s.add(_end(chunk, rank) > instance.steps)
                    else:
                        # On other ranks the chunk can stop being on the rank any time after its start
                        s.add(_end(chunk, rank) >= _start(chunk, rank))

            for rank in collective.ranks():
                # Figure out all addresses plus which ones weill be in the input and output buffers
                addresses = set()
                input_addresses = set()
                output_addresses = set()
                for chunk in collective.chunks():
                    addr = collective.address(chunk)
                    addresses.add(addr)
                    if collective.precondition(rank, chunk):
                        input_addresses.add(addr)
                    if collective.postcondition(rank, chunk):
                        output_addresses.add(addr)
                    # Enforce the address start-end intervals to contain all the chunk start-end intervals
                    s.add(_addr_start(addr, rank) <= _start(chunk, rank))
                    s.add(_addr_end(addr, rank) >= _end(chunk, rank))

                # Statically allocate indices for addresses in the input and output buffers
                next_idx = 0
                for addr in sorted(input_addresses):
                    # Allocate addresses that are both input and output in the output portion
                    if addr not in output_addresses:
                        s.add(_idx(addr, rank) == next_idx)
                        next_idx += 1
                for addr in sorted(output_addresses):
                    s.add(_idx(addr, rank) == next_idx)
                    next_idx += 1

                def conflict(addr1, addr2):
                    s1 = _addr_start(addr1, rank)
                    s2 = _addr_start(addr2, rank)
                    e1 = _addr_end(addr1, rank)
                    e2 = _addr_end(addr2, rank)
                    if not instance.allow_exchange:
                        # Without exhanges the index has to be reserved for the states before and after the interval
                        # (The correctness part of the encoding allows chunks to "hop" from one rank to the next
                        # without overlap)
                        s1 = s1 - 1
                        s2 = s2 - 1
                        e1 = e1 + 1
                        e2 = e2 + 1
                    # There is a conflict if the intervals overlap
                    return And(s1 < e2, s2 < e1)
                
                # Count how many addresses will be in the input and output buffers
                input_size = len(input_addresses)
                output_size = len(output_addresses)
                idx_end = input_size + output_size + instance.extra_memory
                
                # Add constraints for allocating indices for all the addresses just passing through the rank
                for addr in (addresses - input_addresses) - output_addresses:
                    for other in addresses:
                        if other != addr:
                            # If two addresses have the same index they have to have non-conflicting liveness intervals
                            s.add(Implies(_idx(addr, rank) == _idx(other, rank), Not(conflict(addr, other))))
                    # If the address is ever live on this rank require it to be inside the memory limits
                    in_memory = And(0 <= _idx(addr, rank), _idx(addr, rank) < idx_end)
                    s.add(Implies(_addr_start(addr, rank) <= instance.steps, in_memory))

    def solve(self, instance):
        chunked = self.collective.chunk_up(instance.chunks)

        solver = Solver()
        self._encode(solver, instance, chunked)
        if solver.check() == sat:
            model = solver.model()

            # Decode sends from the model
            send_sets = [set() for step in range(instance.steps)]
            for chunk in chunked.chunks():
                addr = chunked.address(chunk)
                for dst in chunked.ranks():
                    for src in self.topology.sources(dst):
                        # Check if the send of chunk from src to dst happens
                        if is_true(model.eval(_send(chunk, src, dst))):
                            # Find which step it happens on (the step before it starts on the destination)
                            step = model.eval(_start(chunk, dst)).as_long() - 1
                            # Filter out "phantom" sends that happen outside the algorithm
                            if 0 <= step and step < instance.steps:
                                send_sets[step].add((addr, src, dst))

            # Store the sends for each step and number of rounds used
            steps = [Step(model.eval(_rounds(i)).as_long(), list(send_sets[i])) for i in range(instance.steps)]

            return Algorithm.make_implementation(self.collective, self.topology, instance, steps)
        else:
            return None

# Prefer using the non-combining dual reduction
PathEncoding = wrap_try_ncd_reduction(PathEncodingBase)
