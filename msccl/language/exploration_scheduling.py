from collections import defaultdict
from math import ceil
from rank_dag import InstructionDAG
from tb_assignment import topo_sort_instrs
from ir import Op, chan_t, rank_t, tbid_t


def assign_balanced_channels(instr_dag: InstructionDAG, num_channels: int, blocked: bool) -> dict[Op, chan_t]:
    toposorted = topo_sort_instrs(instr_dag)
    rank_sends: dict[rank_t, list[Op]] = defaultdict(list)
    for inst in toposorted:
        if inst.is_send():
            rank_sends[inst.rank].append(inst)

    connections: dict[rank_t, list[Op]] = defaultdict(list)

    for send in rank_sends[rank_t(0)]:
        dest = send.recv_peer()
        connections[dest].append(send)

    # assign channels to all the connections leaving rank 0
    channel_assignment: dict[Op, chan_t] = {}
    for sends in connections.values():
        if blocked:
            num_sends = len(sends)
            each_channel = ceil(num_sends / num_channels)
            for i in range(num_sends, step=each_channel):
                block = sends[i:i+each_channel]
                for send in block:
                    channel_assignment[send] = chan_t(i)
        else:
            for i, send in enumerate(sends):
                channel_assignment[send] = chan_t(i % num_channels)


    # copy channel assignments over from rank 0 to the other ranks
    for sends in rank_sends.values():
        for rank0, other in zip(rank_sends[rank_t(0)], sends):
            channel_assignment[other] = channel_assignment[rank0]

    # propagate channel assignments to the corresponding receives
    recv_channel_assignment: dict[Op, chan_t] = {}
    for send in channel_assignment:
        recv: Op = send.recv_match()
        recv_channel_assignment[recv] = channel_assignment[send]

    channel_assignment.update(recv_channel_assignment)

    return channel_assignment



def merge_threadblocks(instr_dag: InstructionDAG, channel_assignment: dict[Op, chan_t], merges: set[chan_t]) -> dict[Op, tbid_t]:
    # this only works because we've assumed all ranks are isomorphic
    tb_groups: dict[str, set[Op]] = defaultdict(set)

    for op in channel_assignment:
        if (chan := channel_assignment[op]) in merges:
            tb_groups[f'sr{chan}'].add(op)
        elif op.is_send():
            tb_groups[f's{chan}'].add(op)
        elif op.is_recv():
            tb_groups[f'r{chan}'].add(op)
        else:
            assert False, "local op somehow made it into the channel assignment??"

    tb_assignment: dict[Op, tbid_t] = {}
    for i, tb in enumerate(tb_groups.values()):
        for op in tb:
            tb_assignment[op] = tbid_t(i + 1) # reserve 0 for all local operations

    return tb_assignment


def apply_manual_schedule(instr_dag: InstructionDAG, channel_assignment: dict[Op, chan_t], tb_assignment: dict[Op, tbid_t]):
    for op in instr_dag.operations.values():
        if op in channel_assignment:
            assert op in tb_assignment
            op.channel = channel_assignment[op]
            op.tb = tb_assignment[op]
        else:
            assert op.is_local()
            op.channel = chan_t(-1)
            op.tb = tbid_t(0)
