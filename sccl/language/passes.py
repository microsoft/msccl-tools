# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
                delete_idx.append(i+1)
    
    delete_operations(ops, tbs, delete_idx)

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

# Within a tb reorder sends before receives if they are independent of each other
# def prioritize_sends(tb):
#     steps = len(tb.ops)
#     for i in range(1, steps):
#         prev = tb.ops[i-1]
#         current = tb.ops[i]
#         if current.inst == Instruction.send and is_receive(prev) and not chunks_overlap(current.src, prev.dst):
#             tb.ops[i-1] = current
#             tb.ops[i] = prev


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
        dep_op = ops[i-1]
        op = ops[i]
        
        # Send's depend on the most recent recv-type instruction
        # Avoid serializing sends that can happen in parallel.
        if op.inst == Instruction.send:
            # If this is a send we depend on the last non-send op
            dep_op_idx = i-1
            while dep_op.inst is Instruction.send and dep_op_idx >= 0:
                dep_op_idx -= 1
                prev_op = ops[dep_op_idx]
            if dep_op_idx == -1:
                continue # No true dependency

        # If we have multiple dependent ops from the same tb keep the one with the highest steps
        depends = op.depends
        tb = dep_op.tb
        if tb not in depends or dep_op.step > depends[tb].step:
            depends[tb] = dep_op




