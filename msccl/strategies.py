# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.instance import Instance
from msccl.path_encoding import PathEncoding
from msccl.rounds_bound import lower_bound_rounds
from msccl.steps_bound import lower_bound_steps

import time
import math
from fractions import Fraction
import itertools
from collections import defaultdict

def _solve_and_log(encoding, instance, logging):
    if logging:
        print(f'Solving instance {instance}... ', end='', flush=True)

    start_time = time.time()
    result = encoding.solve(instance)
    duration = time.time() - start_time
    
    if logging:
        if result != None:
            print(f'synthesized! ({duration:.1f}s)')
        else:
            print(f'unsatisfiable. ({duration:.1f}s)')

    return result

def solve_instance(topology, collective, instance, logging = False):
    encoding = PathEncoding(topology, collective)
    return _solve_and_log(encoding, instance, logging)

def solve_least_steps(topology, collective, initial_steps = 1, base_instance = Instance(None), logging = False):
    if initial_steps < 1:
        raise ValueError('initial_steps must be strictly positive')

    encoding = PathEncoding(topology, collective)

    # Lower bound the number of steps required
    steps_lb = lower_bound_steps(topology, collective)
    if steps_lb == None:
        if logging:
            raise ValueError('The collective is unimplementable in this topology.')
    if logging:
        print(f'Algorithms need at least {steps_lb} steps.')

    num_steps = max(initial_steps, steps_lb)
    if num_steps > steps_lb:
        result = _solve_and_log(encoding, base_instance.set(steps=num_steps), logging)
        if result != None:
            if logging:
                print('Synthesized on initial guess. Checking for fewer steps.')
            while num_steps > steps_lb:
                num_steps -= 1
                maybe_better = _solve_and_log(encoding, base_instance.set(steps=num_steps), logging)
                if maybe_better != None:
                    result = maybe_better
                else:
                    break
            return result
        else:
            num_steps += 1
    
    while True:
        result = _solve_and_log(encoding, base_instance.set(steps=num_steps), logging)
        if result != None:
            return result
        else:
            num_steps += 1

def solve_all_latency_bandwidth_tradeoffs(topology, collective, min_chunks = 1, max_chunks = None, assume_rounds_per_chunk_lb = None, assume_monotonic_feasibility = False, base_instance = Instance(None), logging = False):
    if min_chunks < 1:
        raise ValueError('min_chunks must be strictly positive.')
    if max_chunks != None and max_chunks < min_chunks:
        raise ValueError('max_chunks must be greater or equal to min_chunks.')
    if assume_rounds_per_chunk_lb != None and assume_rounds_per_chunk_lb < 0:
        raise ValueError('assume_rounds_per_chunk_lb must be positive.')

    # Lower bound the number of steps required
    steps_lb = lower_bound_steps(topology, collective)
    if logging:
        print(f'Algorithms need at least {steps_lb} steps.')

    # Lower bound the number of rounds per unit of chunkiness required
    if assume_rounds_per_chunk_lb != None:
        rounds_per_chunk_lb = assume_rounds_per_chunk_lb
        if logging:
            print(f'Assuming algorithms need at least {rounds_per_chunk_lb} rounds per chunk.')
    else:
        rounds_per_chunk_lb = lower_bound_rounds(topology, collective)
        if logging:
            print(f'Algorithms need at least {rounds_per_chunk_lb} rounds per chunk.')

    # Remember for which rounds per chunk fraction a given number of steps will be unsat
    step_rpc_lb = defaultdict(lambda: Fraction(0))

    chunks_iter = range(min_chunks, max_chunks+1) if max_chunks != None else itertools.count(min_chunks)

    algorithms = []
    for chunks in chunks_iter:
        encoding = PathEncoding(topology, collective)
        rounds_lb = math.ceil(rounds_per_chunk_lb * chunks)

        rounds = rounds_lb - 1
        found = False
        while not found:
            rounds += 1
            rpc = Fraction(rounds, chunks)
            # Skip this fraction if a lower number of chunks will have already considered it
            if math.gcd(chunks, rounds) != 1:
                continue
            for steps in range(steps_lb, rounds+1):
                # Skip this number of steps if a previous instance with stricter rounds per chunk already failed
                if assume_monotonic_feasibility and rpc < step_rpc_lb[steps]:
                    continue
                instance = base_instance.set(steps=steps, extra_rounds=rounds - steps, chunks=chunks)
                result = _solve_and_log(encoding, instance, logging=logging)
                if result != None:
                    assert rpc >= step_rpc_lb[steps], 'Monotonic feasibility assumption would have been violated.'
                    found = True
                    yield result
                    break
                else:
                    # Update the rounds per chunk for which this number of steps is not sufficient
                    step_rpc_lb[steps] = max(step_rpc_lb[steps], rpc)
                    if logging and assume_monotonic_feasibility:
                        print(f'Assuming {steps} step algorithms need at least {rpc} rounds per chunk.')
        # Check if a bandwidth optimal algorithm has been found
        if found and rpc <= rounds_per_chunk_lb:
            assert rpc == rounds_per_chunk_lb, 'Rounds per chunk lower bound did not hold.'
            if logging:
                print(f'Bandwidth optimal algorithm found!')
            break
    else:
        if logging:
            print(f'Reached the limit for chunks.')

def _steps(algo):
    return len(algo.steps)

def _rpc(algo):
    return Fraction(_steps(algo) + algo.extra_rounds(), algo.instance.chunks) 

def prune_pareto_optimal(algorithms):
    efficient_algorithms = []
    for i, algo in enumerate(algorithms):
        is_efficient = True
        for j, other in enumerate(algorithms):
            either_worse = _steps(algo) > _steps(other) or _rpc(algo) > _rpc(other)
            neither_better = _steps(algo) >= _steps(other) and _rpc(algo) >= _rpc(other)
            if either_worse and neither_better:
                is_efficient = False
                break
        if is_efficient:
            efficient_algorithms.append(algo)

    return efficient_algorithms
