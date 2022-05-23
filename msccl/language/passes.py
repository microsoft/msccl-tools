# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from msccl.language.ir import *

# Check that there are no cyclic dependencies within a Rank
def check_dependency_cycles(tbs):
    for rank, rank_tbs in enumerate(tbs):
        for tbid, tb in rank_tbs.items():
            for op in tb.ops:
                deps = op.depends
                chain = [op]
                # DFS to check for cycles
                while len(deps) > 0:
                    dep = deps[0]
                    if dep in chain:
                        print(f"Cyclic dependency in rank {rank} threadblock {tbid} at {op}")
                        for op in chain:
                            print("  ", op)
                        sys.exit(1)
                    next_depends = dep.depends
                    if len(next_depends) > 0:
                        chain.append(dep)
                    else:
                        chain = [op]
                    deps = next_depends + deps[1:]


# Check there are no ordering violations between threadblocks across ranks
def check_threadblock_ordering(rank_dag):
    for rank in range(rank_dag.num_ranks):
        for tb in rank_dag.tbs[rank].values():
            prev_steps = {} # tbid -> step of last recv from tbid
            # Check that sends and their corresponding receives between two threadblocks
            # happen in the same order.
            for op_step, op in enumerate(tb.ops):
                if op.is_send():
                    match = op.recv_match
                    if match.is_recv():
                        assert op.dst.rank == match.rank, f"Bug in MSCCLang: Sends don't match receives"

                    other_tbid = match.tb
                    if other_tbid in prev_steps:
                        if match.step <= prev_steps[other_tbid].step:
                            print("Offending Steps", match.step, prev_steps[other_tbid].step)
                            print("Sending tb")
                            for op in tb.ops:
                                print(f'{op.step}: Recv step: {op.recv_match.step if op.is_send() else -1} {op} priority:{(op.chunk_step, op.priority, op.dst.index)}')
                            print("Receiving tb")
                            for op in rank_dag.tbs[match.rank][other_tbid].ops:
                                print(f'{op.step}: {op} priority:{(op.chunk_step, op.priority, op.dst.index)}')
                            assert match.step >  prev_steps[other_tbid].step, f"Rank {op.rank} sends op1 then op2 but {match.rank} receives op2 then op1"
                        
                    prev_steps[other_tbid] = match
