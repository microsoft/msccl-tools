# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from sccl.language.ir import *

# Check that there are no cyclic dependencies within a Rank
def check_dependency_cycles(tbs):
    for tb in tbs.values():
        for op in tb.ops:
            deps = op.depends
            chain = [op]
            # DFS to check for cycles
            while len(deps) > 0:
                dep = deps[0]
                if dep in chain:
                    print("Cyclic dependency", op)
                    for op in chain:
                        print("  ", op)
                    sys.exit(1)
                next_depends = dep.depends
                if len(next_depends) > 0:
                    chain.append(dep)
                else:
                    chain = [op]
                deps = next_depends + deps[1:]


# Check there are no ordering violations between threadblocks across Ranks
def check_threadblock_ordering(rank_dag):
    for rank in range(rank_dag.num_ranks):
        for tb in rank_dag.tbs[rank].values():
            prev_steps = {} # tbid -> step of last recv from tbid

            # Check that sends and their corresponding receives between two threadblocks
            # happen in the same order.
            for op_step, op in enumerate(tb.ops):
                if op.is_send():
                    match = op.match[0]
                    if match.inst == Instruction.recv:
                        assert op.src == match.src and op.dst == match.dst, f"Bug in SCCLang: Sends don't match receives"
                    else:
                        assert op.src == match.src, f"Bug in SCCLang: Sends don't match receives"

                    other_tbid = match.tb
                    if other_tbid in prev_steps:
                        assert match.step >  prev_steps[other_tbid], f"Rank {self.rank} sends op1 then op2 but {match.rank} receives op2 then op1"
                    prev_steps[other_tbid] = match.step


                
                