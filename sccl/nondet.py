# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.algorithm import *
from collections import defaultdict
from dataclasses import dataclass
import random, math

@dataclass
class _BwLimit:
    srcs: set
    dsts: set
    bw: int
    util: int = 0

def nondet_build(topology, collective, instance):
    bools = {}
    def choose_bool(name):
        if not name in bools:
            bools[name] = random.choice([True, False])
        return bools[name]

    ints = {}
    def choose_int(name, a, b):
        if not name in ints:
            ints[name] = random.randint(a, b)
        return ints[name]

    steps = defaultdict(lambda: Step(0, []))

    # Build a tree of scheduled sends for each chunk separately
    for chunk_idx, chunk in enumerate(collective._chunks):
        # Chunks start at time 0 on their precondition ranks
        starts = { rank: 0 for rank in chunk.precondition }
        # Worklist of ranks to be considered as senders
        work = list(chunk.precondition)

        while work:
            src = work.pop()
            # Sends can't happen after the algorithm has ended
            if starts[src] >= instance.steps:
                continue
            for dst in topology.destinations(src):
                # Only one sender for each rank
                if not dst in starts:
                    # Choose if this rank is the sender
                    if choose_bool(f'send{chunk_idx}from{src}to{dst}'):
                        # Choose when the send happens:
                        # -Not earlier than when chunk was received
                        # -Not later than the last step of the algorithm
                        starts[dst] = choose_int(f'start{chunk_idx}on{dst}', starts[src] + 1, instance.steps)
                        steps[starts[dst] - 1].sends.append((chunk_idx, src, dst))
                        work.append(dst)

    # Ignore any time steps when no sends happen
    real_steps = [steps[time] for time in sorted(steps.keys())]

    # Count how long each step takes
    for step in real_steps:
        bw_limits = [_BwLimit(srcs, dsts, bw) for srcs, dsts, bw, _ in topology.bandwidth_constraints()]
        for chunk_idx, src, dst in step.sends:
            for limit in bw_limits:
                if src in limit.srcs and dst in limit.dsts:
                    limit.util += 1
        step.rounds = max(math.ceil(limit.util / limit.bw) for limit in bw_limits)

    # This is the running time of the algorithm (abstractly)
    print(f'Rounds: {sum(step.rounds for step in real_steps)}')
    
    # Check that this was a valid algorithm
    return Algorithm.make_implementation(collective, topology, instance, real_steps)
