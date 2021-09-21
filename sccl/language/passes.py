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
    for i in range(0, len(ops)):
        ops[i].depends = []

def update_slot_dependency(ops):
    for i in range(1, len(ops)):
        ops[i].depends.append(ops[i-1])


# TODO: will only work for a very specific pattern...
# Reorders rrs to occur before rrc in the same block
def reorder_rrs_rrc(tb):
    ops = tb.ops
    last_rrc = -1
    for i in range(0, len(ops)):
        op = ops[i]
        if op.inst == Instruction.recv_reduce_copy and last_rrc == -1 and ops[last_rrc].src.index != ops[i].src.index:
            last_rrc = i
        # elif op.inst == Instruction.recv_reduce_send and last_rrc > -1:
        #     # Swap the rrc and rrs so that the rrs comes first.
        #     temp = ops[last_rrc]
        #     ops[last_rrc] = ops[i]
        #     ops[i] = temp
        #     last_rrc += 1
        #     print(f'Reordered {ops[last_rrc-1]} and {ops[i]}')


