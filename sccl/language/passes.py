# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.language.ir import *

def same_tb(op1, op2):
    return op1.tb == op2.tb

def same_count(op1, op2):
    return op1.cnt() == op2.cnt()

def delete_operations(ops, tbs, delete_idx):
    delete_idx.sort(reverse=True)
    # Delete the ops
    for i in delete_idx:
        ops[i].inst = Instruction.delete
        del ops[i]


# Given the set of operations that operate over a particular slot (rank, buffer, idx) fixed
# Try and replace operations with pipelined ops like receive copy send (rcs)
# or receive reduce send (rrs) and receive reduce copy send (rrcs)
# TODO: Only works if there are no multi chunk sends!!!!!!
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
                delete_idx.append(i+1)
    
    delete_operations(ops, tbs, delete_idx)

    # # Update the depends of ops - always depend on the ops ahead 
    # for i in range(1, len(ops)):
    #     ops[i].depends = [ops[i-1]]


def rrcs_rrs(ops, tbs):
    delete_idx = []
    if len(ops) >= 3:
        for i in range(0, len(ops)-2):
            if ops[i].inst == Instruction.recv_reduce_copy and ops[i+1].inst == Instruction.send and ops[i+2].inst == Instruction.recv and same_tb(ops[i], ops[i+1]) and same_count(ops[i], ops[i+1]):
                ops[i].inst = Instruction.recv_reduce_send
                ops[i].dst = ops[i+1].dst
                delete_idx.append(i+1)

    if len(ops) >= 2:
        for i in range(0, len(ops)-1):
            if ops[i].inst == Instruction.recv_reduce_copy and ops[i+1].inst == Instruction.send and same_tb(ops[i], ops[i+1]) and same_count(ops[i], ops[i+1]):
                ops[i].inst = Instruction.recv_reduce_copy_send
                ops[i].dst = ops[i+1].dst
                delete_idx.append(i+1)
    
    delete_operations(ops, tbs, delete_idx)

    # # Update the depends of ops - always depend on the ops ahead 
    # for i in range(1, len(ops)):
    #     ops[i].depends = [ops[i-1]]

# Performs the deletion of operations that are marked delete
def delete_pass(tb):
    steps = len(tb.ops) - 1
    for s in range(steps, -1, -1):
        if tb.ops[s].inst == Instruction.delete:
            del tb.ops[s]

def clear_dependency(ops):
    for op in ops:
        op.depends = {}

def update_slot_dependency(ops):           
    for i in range(1, len(ops)):
        last_op = ops[i-1]
        op = ops[i]
        # If we have multiple dependent ops from the same tb keep the one with the highest steps
        tb = last_op.tb
        depends = op.depends
        if tb not in depends or last_op.step > depends[tb].step:
                depends[tb] = last_op



