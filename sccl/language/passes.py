# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from sccl.language.ir import *

def same_tb(op1, op2):
    return op1.tb == op2.tb

def same_count(op1, op2):
    return op1.cnt() == op2.cnt()

def is_receive(op):
    return op.inst == Instruction.recv or op.inst == Instruction.recv_reduce_copy

def chunks_overlap(chunk1, chunk2):
    if chunk1.buffer == chunk2.buffer:
        if chunk1.index < chunk2.index:
            lower_chunk = chunk1
            upper_chunk = chunk2
        else:
            lower_chunk = chunk2
            upper_chunk = chunk1
        if lower_chunk.index <= upper_chunk.index and (lower_chunk.index + lower_chunk.size -1) >= upper_chunk.index:
            return True
    return False

def delete_operations(ops, tbs, delete_idx):
    delete_idx.sort(reverse=True)
    # Delete the ops
    for i in delete_idx:
        ops[i].inst = Instruction.delete
        del ops[i]

def remove_op(ops, i):
    ops[i].next.remove(ops[i+1])
    ops[i].next = ops[i].next + ops[i+1].next
    for o in ops[i+1].next:
        o.prev.remove(ops[i+1])
        o.prev.append(ops[i])

# Given the set of operations that operate over a particular slot (rank, buffer, idx) fixed
# Try and replace operations with pipelined ops like receive copy send (rcs)
# or receive reduce send (rrs) and receive reduce copy send (rrcs)
# Rules:
# recv-copy-send 
# recv(src, sbuf, si, _, _, _ ) send(_, _, _, dst, dbuf, di) -> recv_copy_send(src, sbuf, si, dst, dbuf, di)
def rcs(ops, tbs):
    delete_idx = [] # Which index to delete from this ops
    if len(ops) >= 2:
        for i in range(0, len(ops)-1):
            if ops[i].inst == Instruction.recv and ops[i+1].inst == Instruction.send and same_tb(ops[i], ops[i+1]) and same_count(ops[i], ops[i+1]):
                ops[i].inst = Instruction.recv_copy_send
                ops[i].dst = ops[i+1].dst
                remove_op(ops, i)
                delete_idx.append(i+1)
    
    delete_operations(ops, tbs, delete_idx)

def rrcs_rrs(ops, tbs):
    delete_idx = []

    # RRC/S -> RRS
    if len(ops) >= 3:
        for i in range(0, len(ops)-2):
            if ops[i].inst == Instruction.recv_reduce_copy and ops[i+1].inst == Instruction.send and ops[i+2].inst == Instruction.recv and same_tb(ops[i], ops[i+1]) and same_count(ops[i], ops[i+1]):
                ops[i].inst = Instruction.recv_reduce_send
                ops[i].dst = ops[i+1].dst
                remove_op(ops, i)
                delete_idx.append(i+1)

    # RRC/S -> RRCS
    if len(ops) >= 2:
        for i in range(0, len(ops)-1):
            if ops[i].inst == Instruction.recv_reduce_copy and ops[i+1].inst == Instruction.send and same_tb(ops[i], ops[i+1]) and same_count(ops[i], ops[i+1]):
                ops[i].inst = Instruction.recv_reduce_copy_send
                ops[i].dst = ops[i+1].dst
                remove_op(ops, i)
                delete_idx.append(i+1)

    
    delete_operations(ops, tbs, delete_idx)

# Performs the deletion of operations that are marked delete
def delete_pass(tb):
    steps = len(tb.ops) - 1
    for s in range(steps, -1, -1):
        if tb.ops[s].inst == Instruction.delete:
            del tb.ops[s]

def clear_dependency(ops):
    for op in ops:
        op.depends = {}

def update_slot_dependency(slot, ops): 
    for i in range(1, len(ops)):
        dep_op = ops[i-1]
        op = ops[i]

        # Send's depend on the most recent recv-type instruction
        # Avoid serializing sends that can happen in parallel.
        if op.inst == Instruction.send:
            # If this is a send we depend on the last non-send op
            dep_op_idx = i-1
            while dep_op.inst is Instruction.send and dep_op_idx > 0:
                dep_op_idx -= 1
                dep_op = ops[dep_op_idx]
            if dep_op.inst is Instruction.send:
                continue # No true dependency
            dep_ops = [dep_op]
        # Receive and reduce instructions depend on the previous receive/reduce instructions
        # or all parallel sends that happen before it       
        else:
            dep_ops = [dep_op]
            dep_op_idx = i-1
            while dep_op.inst is Instruction.send and dep_op_idx > 0:
                dep_op_idx -= 1
                dep_op = ops[dep_op_idx]
                if dep_op.inst is Instruction.send:
                    dep_ops.append(dep_op)

        # If we have multiple dependent ops from the same tb keep the one with the highest steps
        depends = op.depends
        for dep_op in dep_ops:
            tb = dep_op.tb
            if tb not in depends or dep_op.step > depends[tb].step:
                depends[tb] = dep_op

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
def check_threadblock_ordering(tbs, ranks):
    for tb in tbs.values():
        other_tbs = {} # TBs this TB communicates with (r, tb) -> step of last op
        # Gather all unique TBs this TB communicates with
        rerun = False
        for op in tb.ops:
            for prev in op.prev:
                other_tbs[(prev.rank, prev.tb)] = (-1, -1)
            for next in op.next:
                other_tbs[(next.rank, next.tb)] = (-1, -1)
        # Check that the ordering of operations between threadblocks is consistent
        for op_step, op in enumerate(tb.ops):
            for next in op.next:
                next_step = ranks[next.rank].tbs[next.tb].ops.index(next)
                other_tb_step, prev_tb_step = other_tbs[(next.rank, next.tb)]
                if other_tb_step > next_step:
                    print("Ordering problem", other_tb_step, next_step, op)
                    sys.exit(1)
                other_tbs[(next.rank, next.tb)] = (next_step, op.step)
                