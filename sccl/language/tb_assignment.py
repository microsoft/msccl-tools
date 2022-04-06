# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from enum import Enum
import heapq

from sccl.language.ir import *
from sccl.language.rank_dag import *


def _verify_tb_op_compatible(tb, op):
    s = op.dst.rank if op.is_send() else -1
    r = op.src.rank if op.is_recv() else -1
        
    sends_ok = tb.send == s or s == -1 or tb.send == -1
    recvs_ok = tb.recv == r or r == -1 or tb.recv == -1
    channel_ok = tb.channel == op.channel or tb.channel == -1 or op.channel == -1
    return sends_ok and recvs_ok and channel_ok

# Manual threadblock, channel assignment
def manual_assign_tbs(rank_dag):
    instrs = topo_sort_instrs(rank_dag)
    for op in instrs:
        
        rank = op.rank
        tbid = op.tb
        if tbid not in rank_dag.tbs[rank]:
            rank_dag.tbs[rank][tbid] = Threadblock()
        tb = rank_dag.tbs[rank][tbid]
        if _verify_tb_op_compatible(tb, op):
            tb.ops.append(op)
            tb.channel = op.channel if op.channel != -1 else 0
            tb.send = op.dst.rank if op.is_send() else tb.send
            tb.recv = op.src.rank if op.is_recv() else tb.recv
            op.step = len(tb.ops)-1
            rank_dag.num_channels[rank] = max(op.channel+1, rank_dag.num_channels[rank] )
        else:
            raise Exception(f"Illegal threadblock assignment. Trying to add {op} to threadblock {tbid}\n" \
                f"Threadblock {tbid} send:{tb.send} recv:{tb.recv} channel:{tb.channel}\n" \
                f"Operation send:{op.dst.rank if op.is_send() else -1} recv:{op.dst.rank if op.is_recv() else -1} channel:{op.channel}")

def _get_tb_options(mapping, send, recv, channel, num_tbs):
    options = []
    for tbid, tb in mapping.items():
        tb_s = tb.send
        tb_r = tb.recv
        tb_c = tb.channel
        sender_ok = send == -1 or tb_s == send
        receiver_ok = recv == -1 or tb_r == recv
        channel_ok = channel == -1 or channel == tb_c
        # For correctness - if one of the peer's channels is already allocated we must use it.
        if channel_ok and ((tb_s == send and send != -1) or (tb_r == recv and recv != -1)):
            return [tbid]
        if sender_ok and receiver_ok and channel_ok:
             options.append(tbid)
    return options

def auto_assign_tbs(rank_dag):
    instrs = topo_sort_instrs(rank_dag)
    channel_assignment(instrs, rank_dag)
    rank_tbids = [0] * rank_dag.num_ranks
    current_tb_step = []
    for rank_tbs in rank_dag.tbs:
        current_tb_step.append({})

    for op in instrs:
        rank = op.rank
        s = op.send_peer()
        r = op.recv_peer()
        channel = 0 if op.channel == -1 else op.channel
        # Get all possible TBs this can be mapped to
        tb_options = _get_tb_options(rank_dag.tbs[rank], s, r, channel, rank_tbids[rank])
        if len(tb_options) == 0: # If there are no options, create a new threadblock
            tbid = rank_tbids[rank]
            # if op.channel == -1:
            #     print(op.channel, op)
            rank_dag.tbs[rank][tbid] = Threadblock(send=s, recv=r, channel=channel)
            # rank_tb_assignments[rank][(s,r,channel)] = tbid
            rank_tbids[rank] += 1
        else: 
            tbid = tb_options[0]
            for tbid_opt in tb_options:
                if current_tb_step[rank][tbid_opt] < current_tb_step[rank][tbid] and _verify_tb_op_compatible(rank_dag.tbs[rank][tbid], op):
                    tbid = tbid_opt
            # if op.chunk_step < current_tb_step[rank][tbid]:
            #     tbid = rank_tbids[rank]
            #     rank_dag.tbs[rank][tbid] = Threadblock(send=s, recv=r, channel=channel)
            #     rank_tbids[rank] += 1

        tb = rank_dag.tbs[rank][tbid]
        assert _verify_tb_op_compatible(tb, op), f"Failing: Operations uses channel {op.channel}, send:{s} recv:{r} {op}\n" \
                f"Threadblock uses send:{tb.send} recv:{tb.recv} channel:{tb.channel}"

        rank_dag.num_channels[rank] = max(rank_dag.num_channels[rank], channel+1)

        tb.ops.append(op)
        tb.send = op.dst.rank if op.is_send() else tb.send
        tb.recv = op.src.rank if op.is_recv() else tb.recv
        
        op.step = len(tb.ops)-1
        op.tb = tbid
        current_tb_step[rank][tbid] = op.chunk_step

# Topologically orders instructions so that (1): Sends occur before their receives
# (2): Dependent instructions occur before 
def topo_sort_instrs(rank_dag):
    visited = set()
    ops = []
    ordered = []
    for slot, op in rank_dag.operations.items():
        if op.inst == Instruction.start:
            visited.add(op)
            for o in op.next:
                if o.inst == Instruction.send or o.inst == Instruction.copy:
                    heapq.heappush(ops, ((o.chunk_step, -o.priority, o.dst.index), o))

    while len(ops) > 0:
        _, op = heapq.heappop(ops)
        if op not in visited:
            rmatch = op.recv_match

            # Delay scheduling the send until the receive is ready
            if rmatch is not None and not all([o in visited for o in rmatch.prev]):
                heapq.heappush(ops, ((rmatch.chunk_step-1, -rmatch.priority+1, op.dst.index), op))
            else:
                ordered.append(op)
                visited.add(op)
                
                # Add a matching receive if one exists
                if rmatch is not None : 
                    heapq.heappush(ops, ((rmatch.chunk_step, -op.priority+1, rmatch.dst.index), rmatch))
                # Add other operation that has its dependencies satisfied
                for o in op.next:
                    if all([x in visited for x in o.prev]):
                        heapq.heappush(ops, ((o.chunk_step, -o.priority, o.dst.index), o))

    return ordered

def channel_assignment(instrs, rank_dag):
    def f():
        return set()
    # Naive - schedule everything onto channel 0
    # First handle flows - if an instruction at Rx is fused Rw->Rx->Ry and takes c
    # Then flow Rw->Rx->Rz must be ib a different channel c' where c!=c'
    rank2sendch = [defaultdict(f) for _ in range(rank_dag.num_ranks)]
    rank2recvch = [defaultdict(f) for _ in range(rank_dag.num_ranks)]

    # DFS through the InstructionDAG identifying flows
    # TODO: This could probably be merged with the metadata collection...
    def valid_send_ch(sender, receiver, ch):
        return ch not in rank2sendch[sender][receiver]
    def valid_recv_ch(sender, receiver, ch):
        return ch not in rank2recvch[receiver][sender]

    flows = []
    flow_channels = []

    def dfs(op, channel, f):
        if op.is_local():
            op.channel = 0
        elif op.is_send():
            match = op.recv_match
            sender = op.rank
            receiver = match.rank
            # Find the first usable channel
            ch = channel
            while not(valid_send_ch(sender, receiver, ch) and valid_recv_ch(sender, receiver, ch)):
                ch += 1
            # If not a fused op use the first possible channel (send, recv/rrc)
            if not match.is_fused():
                # Check if this flow exists
                f.append(op.rank)
                f.append(match.rank)
                if f in flows:
                    ch = flow_channels[flows.index(f)]
                else:
                    flows.append(f)
                    flow_channels.append(ch)
                op.channel = ch
                match.channel = ch
                rank2sendch[sender][receiver].add(ch)
                rank2recvch[receiver][sender].add(ch)
            else:
                f.append(op.rank)
                dfs(match, ch, f)
                ch = match.channel
                op.channel = ch
                rank2sendch[sender][receiver].add(ch)

    # Assign channels to flows
    for op in instrs:
        if op.inst == Instruction.send and op.recv_match.is_fused():
            dfs(op, 0, [])