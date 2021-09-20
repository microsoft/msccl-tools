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

    # Update the depends of ops - always depend on the ops ahead 
    for i in range(1, len(ops)):
        ops[i].depends = [ops[i-1]]


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

    # Update the depends of ops - always depend on the ops ahead 
    for i in range(1, len(ops)):
        ops[i].depends = [ops[i-1]]

# Performs the deletion of operations that are marked delete
def delete_pass(tb):
    steps = len(tb.ops) - 1
    for s in range(steps, -1, -1):
        if tb.ops[s].inst == Instruction.delete:
            del tb.ops[s]

# TODO: Fix this
# Matching pattern rrc, s -> rrs rrc with multi count operations
def multicount_rrcs(tb):
    ops = tb.ops
    last_rrc = -1
    for i in range(0, len(ops)):
        op = ops[i]
        if op.inst == Instruction.recv_reduce_copy and last_rrc == -1:
            last_rrc = i
        elif op.inst == Instruction.recv_reduce_send and last_rrc > -1:
            temp = ops[last_rrc]
            ops[last_rrc] = ops[i]
            ops[i] = temp
            last_rrc += 1


            # print(op1)
            # print(op2)
            # op1_start = op1.dst.index
            # op1_end = op1_start + op1.dst.size
            # op2_start = op2.src.index
            # op2_end = op2_start + op2.dst.size
            # Check that the send operates on a subset of the rrc chunks
            # if op2_start >= op1_start and op2_end <= op1_end:
            #     new_rrc_size = op1.dst.size - op2.src.size
            #     rrc_first = op2_start > op1_start # Determine if the first half chunks are for rrc or rrs
            #     split_index = op2_start if rrc_first else op2_end

            #     if rrc_first:
            #         # Split into rrc and rrs
            #         new_op1_src = ChunkRef(op1.src.buffer, op1.src.index, new_rrc_size)
            #         new_op1_dst = ChunkRef(op1.dst.buffer, op1.dst.index, new_rrc_size)
                    
            #     else:
            #         # Split into rrs and rrc
            #         new_op1_src = ChunkRef(op1.src.buffer, op1.src.index, new_rrc_size)
            #         new_op1_dst = ChunkRef(op1.dst.buffer, split_index, new_rrc_size)
                
            #     op2.inst = Instruction.recv_reduce_send
            #     op1.src = new_op1_src
            #     op1.dst = new_op1_dst
            # print("   ", op1)
            # print("   ", op2)


