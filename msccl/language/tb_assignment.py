# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from enum import Enum
from functools import wraps
import heapq
from itertools import cycle, islice, permutations, product
from typing import Dict, Generator, Tuple, NewType, TypeVar

from msccl.language.ir import *
from msccl.language.rank_dag import *

from copy import deepcopy

from random import random, seed, shuffle



def _verify_tb_op_compatible(tb: Threadblock, op: Op):
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

def _get_tb_options(mapping: dict[tbid_t, Threadblock], send: rank_t, recv: rank_t, channel: chan_t, num_tbs: int):
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
    channels = channel_assignment(instrs, rank_dag)
    for op, ch in zip(instrs, channels):
        op.channel = ch

    
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
            rank_dag.tbs[rank][tbid] = Threadblock(send=s, recv=r, channel=channel)
            rank_tbids[rank] += 1
        else:
            tbid = tb_options[0]
            for tbid_opt in tb_options:
                if current_tb_step[rank][tbid_opt] < current_tb_step[rank][tbid] and _verify_tb_op_compatible(rank_dag.tbs[rank][tbid], op):
                    tbid = tbid_opt
        
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


def reorder_sends_and_recvs(rank_dag):
    """make certain the sends and receives between a pair of GPUs are ordered consistently"""
    
    # get receive peer tb for each tb
    listening_on: Dict[int, Dict[Tuple[int, int], int]] = defaultdict(dict) # int -> ((int, int) -> int) : rank -> (channel, rank_recv_peer) -> tbid listening on that channel
    for rank, rank_tbs in enumerate(rank_dag.tbs):
        for tbid in rank_tbs:
            channel = rank_tbs[tbid].channel
            rank_recv_peer = rank_tbs[tbid].recv

            if -1 not in (channel, rank_recv_peer):
                listening_on[rank][channel, rank_recv_peer] = tbid

    tb_recv_peer = {} # (int, int) -> (int, int) : (rank, tbid) that's sending -> (rank, tbid) that's listening
    for rank, rank_tbs in enumerate(rank_dag.tbs):
        for tbid in rank_tbs:
            tb = rank_tbs[tbid]
            if tb.send == -1:
                continue
            tb_recv_peer[rank, tbid] = (tb.send, listening_on[tb.send][tb.channel, rank])

    # for each sequence of sends in a tb, iterate over the matching sequence of recvs and add dependences to enforce the ordering
    for send_rank, send_tbid in tb_recv_peer:
        # recv_rank, recv_tbid = tb_recv_peer[send_rank, send_tbid]
        send_sequence = list(filter(lambda op: op.inst == Instruction.send, rank_dag.tbs[send_rank][send_tbid].ops))

        for i in send_sequence:
            i.next = set(i.next)
            i.prev = set(i.prev)
            i.recv_match.next = set(i.recv_match.next)
            i.recv_match.prev = set(i.recv_match.prev)

        for p, n in zip(send_sequence, send_sequence[1:]):
            p.next.add(n)
            n.prev.add(p)
            p.recv_match.next.add(n.recv_match)
            n.recv_match.prev.add(p.recv_match)

        
    # TODO: find a better way to do this, maybe
    new_ordered = topo_sort_instrs(rank_dag) # if this fails, the previous step probably created a cycle :)
    for rank, rank_tbs in enumerate(rank_dag.tbs):
        for tbid in rank_tbs:
            rank_tbs[tbid].ops.sort(key=new_ordered.index)
            for i, op in enumerate(rank_tbs[tbid].ops):
                op.step = i

    rank_dag.convert_set_list()

# Topologically orders instructions so that (1): Sends occur before their receives
# (2): Dependent instructions occur before 
def topo_sort_instrs(rank_dag: InstructionDAG):
    def priority(op: Op):
        return ((op.chunk_step, -op.priority, op.dst.index))

    visited: set[Op] = set()
    ops: list[tuple[tuple[int, int, int], Op]] = []
    ordered: list[Op] = []
    for slot, op in rank_dag.operations.items():
        if op.inst == Instruction.start:
            visited.add(op)
            for o in op.next:
                if o.inst == Instruction.send or o.inst == Instruction.copy:
                    heapq.heappush(ops, (priority(o), o))

    while len(ops) > 0:
        _, op = heapq.heappop(ops)
        if op not in visited:
            rmatch = op.recv_match
            ordered.append(op)
            visited.add(op)
            
            # Add a matching receive if one exists and its dependencies are satisfied
            if rmatch is not None and all([x in visited for x in rmatch.prev]): 
                heapq.heappush(ops, (priority(rmatch), rmatch))
            # Add other operation that have dependencies satisfied
            for o in op.next:
                if all([x in visited for x in o.prev]):
                    heapq.heappush(ops, (priority(o), o))
    return ordered


T = TypeVar('T')
def all_partitions(s: set[T], maxlen=32, random_seed=-1):
    def tuple_partitions(elm: set[T]):
        if len(elm) < 2:
            return

        patterns = list(range(1, 1 << (len(elm) - 1)))
        if random_seed != -1:
            seed(random_seed)
            shuffle(patterns)

        for pattern in patterns:
            pat = pattern * 2
            partition: tuple[set[T], set[T]] = (set(), set())

            for e in elm:
                partition[pat % 2].add(e)
                pat //= 2

            yield partition

    def partitions_with_fixed(fixed: list[set[T]], suffix: set[T]):
        yield fixed + [suffix]

        if len(fixed) == maxlen - 1:
            return

        for suffix1, suffix2 in tuple_partitions(suffix):
            yield from partitions_with_fixed(fixed + [suffix1], suffix2)

    yield from partitions_with_fixed([], s)


instr_t = NewType('instr_t', int)

def manual_channel_enumeration(instrs: list[Op], maxchannels=32, seed=-1):
    sr_pairs: set[tuple[instr_t, instr_t]] = set()

    for i, op in enumerate(instrs):
        if op.inst == Instruction.send:
            match: Op = op.recv_match # type: ignore
            sr_pairs.add((instr_t(i), instr_t(instrs.index(match))))

    for partition in all_partitions(sr_pairs, maxlen=maxchannels, random_seed=seed):
        assignment: dict[instr_t, chan_t] = defaultdict(lambda: chan_t(0))
        for i, part in enumerate(partition):
            for s, r in part:
                assignment[s] = chan_t(i)
                assignment[r] = chan_t(i)

        yield assignment



def manual_threadblock_coloring(instrs: list[Op], channel_assignment: dict[instr_t, chan_t], maxthreadblocks=80):
    import networkx as nx # type: ignore

    def color_rank_instrs(vertices: set[instr_t]) -> dict[instr_t, int]:
        sr_instrs: set[instr_t] = set(filter(lambda i: not instrs[i].is_local(), vertices))
        
        # build the graph
        graph = nx.Graph()
        graph.add_nodes_from(vertices)

        must_be_identified = lambda i, j: (instrs[i].is_send() and instrs[j].is_send() and instrs[i].dst == instrs[j].dst) or (instrs[i].is_recv() and instrs[j].is_recv() and instrs[i].src == instrs[j].src)
        graph = nx.quotient_graph(graph, must_be_identified)

        # connect two instructions by an edge if they cannot be in the same threadblock
        for i in sr_instrs:
            for j in sr_instrs:
                # TODO: comment this out when threadblocks get two channels
                if channel_assignment[i] != channel_assignment[j]:
                    graph.add_edge(i, j, reason='ch')
                if instrs[i].is_send() and instrs[j].is_send() and (instrs[i].dst != instrs[j].dst or channel_assignment[i] != channel_assignment[j]):
                    graph.add_edge(i, j, reason='rank')
                if instrs[i].is_recv() and instrs[j].is_recv() and (instrs[i].src != instrs[j].src or channel_assignment[i] != channel_assignment[j]):
                    graph.add_edge(i, j, reason='rank')

        # either produce a single coloring or somehow enumerate colors, or search the space of colors...
        return nx.coloring.greedy_color(graph)



    group_by_rank: dict[rank_t, set[instr_t]] = {}
    for i, instr in enumerate(instrs):
        group_by_rank[instr.rank].add(instr_t(i))

    # now do the coloring on each rank
    tb_assignment: dict[instr_t, int] = {}
    for rank in group_by_rank:
        tb_assignment.update(color_rank_instrs(group_by_rank[rank]))

    return tb_assignment


@dataclass
class schedule:
    tb_info: dict[rank_t, dict[tbid_t, Threadblock]]
    tbids: dict[rank_t, int]
    tb_assignment: dict[instr_t, tbid_t]
    instrs: list[Op]
    ch_assignment: dict[instr_t, chan_t]

def manual_threadblock_enumeration(instrs: list[Op], channel_assignment: dict[instr_t, chan_t], maxtb=80):
    print('starting enumeration...')

    def subsets_of_size(s: set[T], n: int) -> Generator[set[T], None, None]:
        if len(s) < n:
            return

        if n == 0:
            yield set()
        
        for elm in s:
            for subset in subsets_of_size(s - {elm}, n - 1):
                yield subset | {elm}
            yield from subsets_of_size(s - {elm}, n)
                

    def assign_tb_to_sr(rank: set[instr_t]):
        """Returns a generator that iterates over all legal threadblock assignments to *just the sends and receives* in the instruction sequence"""

        # TODO: this isn't quite right: first of all, it allows packing together a send and a receive with different channels
        # TODO: second of all, *not all sends and receives have to be packed together, they can go on their own channel just because. This doesn't do that. 
        # TODO: that last point could be amended by e.g. first partitioning the smaller set and then doing the monomorphism search on the first partition, 
        # TODO: putting all the other partitions on new threadblocks. but the combinatorics of that seem painful so I'm leaving it like this for the moment...

        tb_assignment: dict[instr_t, tbid_t] = {}
        
        # we can immediately assign all nonlocal instructions to a unique threadblock; the question is where do we overlap?
        sends = {r for r in rank if instrs[r].is_send()}
        recvs = {r for r in rank if instrs[r].is_recv()}


        # each group must go on a unique threadblock
        grouped_sends: dict[tuple[rank_t, chan_t], set[instr_t]] = defaultdict(set)
        grouped_recvs: dict[tuple[rank_t, chan_t], set[instr_t]] = defaultdict(set)

        for i in sends:
            recv_match: Op = instrs[i].recv_match # type: ignore
            grouped_sends[recv_match.rank, channel_assignment[i]].add(i)

        for i in recvs:
            send_match: Op = instrs[i].send_match # type: ignore
            grouped_recvs[send_match.rank, channel_assignment[i]].add(i)

        # these should technically be set[set[...]] not list[set[T]] but sets aren't hashable and I'm too lazy to mess with frozenset rn
        send_groups = list(grouped_sends.values())
        recv_groups = list(grouped_recvs.values())

        print(send_groups)
        print(recv_groups)

        # now we need to pick an association; i.e. an injective map A -> B where |A| < |B|
        A_group, B_group = min(send_groups, recv_groups, key=len), max(recv_groups, send_groups, key=len)
        A = set(range(len(A_group)))
        B = set(range(len(B_group)))

        print(f'|A|, |B| = {len(A), len(B)}, {A_group}, {B_group}')

        # assignment for B is fixed since its the larger one
        for tbid, grp in zip(B, B_group):
            for instr in grp:
                tb_assignment[instr] = tbid_t(tbid)

        print(f'Assigned B: {tb_assignment}')


        # "to enumerate monomorphisms, first enumerate cokernels and then enumerate isomorphisms"
        for coker in subsets_of_size(B, len(B) - len(A)):
            im = sorted(list(B - coker))
            # print(f'cokernel = {coker} (so image = {im})')
            for iso in permutations(A):
                assert len(iso) == len(im)
                # print(f'Using iso {iso}')
                for dom, codom in zip(iso, im):
                    for elm in A_group[dom]:
                        tb_assignment[elm] = tbid_t(codom)
                        # print(f'\t{elm} => {codom}')

                # print(f'Yielding {tb_assignment}')
                # input()
                yield tb_assignment


    rank_instrs: dict[rank_t, set[instr_t]] = defaultdict(set)
    for i, instr in enumerate(instrs):
        rank_instrs[instr.rank].add(instr_t(i))

    def assign_tb_to_rank(rank: set[instr_t]):
        print('Assigning to rank...')
        local_instrs = {r for r in rank if instrs[r].is_local()}

        for partial_assignment in assign_tb_to_sr(rank):
            # print(f'{rank} = {local_instrs} U {set(partial_assignment.keys())}')
            assert not (missing := rank - local_instrs - set(partial_assignment.keys())), missing
            local_assignment = {}
            local_assignment.update(partial_assignment)
            # print(f'Partial assignment for s/r: {partial_assignment}')
            for partition in all_partitions(local_instrs, maxtb - len(partial_assignment.values())):
                for i, piece in enumerate(partition):
                    for instr in piece:
                        local_assignment[instr] = tbid_t(i)
                yield local_assignment

    all_assignments = []
    print(rank_instrs)
    
    for _, rank in rank_instrs.items():
        all_assignments.append(list(assign_tb_to_rank(rank)))

    print(f'Assigned for each rank! Computing all combinations...({list(map(len, all_assignments))})')
    for rank_assignments in islice(zip(*map(cycle, all_assignments)), max(map(len, all_assignments))):
        tb_assignment: dict[instr_t, tbid_t] = {}
        for rank_assignment in rank_assignments:
            tb_assignment.update(rank_assignment)
        yield schedule(tb_info={}, tbids={}, tb_assignment=tb_assignment, instrs=[], ch_assignment=channel_assignment)




def assign_tbs(rank_dag: InstructionDAG, instrs: list[Op], channel_assignment: dict[instr_t, chan_t], tb_limit=80, single=False):
    def assign_op(op: instr_t, st: schedule):
        rank = st.instrs[op].rank
        s = st.instrs[op].send_peer()
        r = st.instrs[op].recv_peer()
        channel = chan_t(0) if channel_assignment[op] == -1 else channel_assignment[op]

        tb_options = _get_tb_options(st.tb_info[rank], s, r, channel, st.tbids[rank])
        if len(tb_options) == 0: # if there are no options, allocate a new threadblock
            tbid = tbid_t(st.tbids[rank])
            # print(f"Allocating new threablock: {tbid} ({st.tbids})")
            if tbid >= tb_limit: # ran out of threadblocks on this rank, nothing good is going to come of this state
                print(f"HIT TB LIMIT: {tbid}")
            else:
                st.tb_info[rank][tbid] = Threadblock(send=s, recv=r, channel=channel)
                st.tbids[rank] += 1
                tb_options.append(tbid)

        # now that we have a list of options, enumerate them in the possible new states
        for tbid in tb_options:
            new_st = deepcopy(st)
            
            tb = new_st.tb_info[rank][tbid]
            assert _verify_tb_op_compatible(tb, st.instrs[op]), f"Failing: Operations uses channel {channel_assignment[op]}, send:{s} recv:{r} {op}\n" \
                f"Threadblock uses send:{tb.send} recv:{tb.recv} channel:{tb.channel}"

            tb.ops.append(st.instrs[op])
            tb.send = st.instrs[op].dst.rank if st.instrs[op].is_send() else tb.send
            tb.recv = st.instrs[op].src.rank if st.instrs[op].is_recv() else tb.recv

            new_st.tb_assignment[op] = tbid

            yield new_st

    
    states: list[schedule] = [schedule(tb_info=defaultdict(dict), tbids=defaultdict(int), tb_assignment={}, instrs=instrs, ch_assignment=channel_assignment)]
    for i, op in enumerate(instrs):
        new_states: list[schedule] = []
        for st in states:
            if single:
                new_states += [next(assign_op(instr_t(i), st))]
            else:
                new_states += list(assign_op(instr_t(i), st))
        states = new_states

    return states


def enumerate_all_schedules(rank_dag: InstructionDAG, random_seed=0, count=100, maxchannels=32, maxtb=80):
    print(f"Threadblock limit: {maxtb} (due to instancing)")
    instrs = topo_sort_instrs(rank_dag)
    schedules: list[schedule] = []

    for channel_assignment in manual_channel_enumeration(instrs, maxchannels=maxchannels, seed=random_seed):
        schedules += assign_tbs(rank_dag, instrs, channel_assignment, tb_limit=maxtb)
        print(f'{len(schedules)}/{count}')
        if len(schedules) >= count:
            break
    return schedules[:count]



def enumerate_schedules_fixch(rank_dag: InstructionDAG, random_seed=0, count=100, maxchannels=32, maxtb=80):
    skip = 1

    instrs = topo_sort_instrs(rank_dag)
    
    channels = manual_channel_enumeration(instrs, maxchannels=maxchannels, seed=random_seed)
    list(zip(range(skip), channels)) # >:)
    channel_assignment = next(channels)

    # schedules = assign_tbs(rank_dag, instrs, channel_assignment, tb_limit=maxtb)
    schedules = list(manual_threadblock_enumeration(instrs, channel_assignment, maxtb))
    print(f'{len(schedules)} generated total')
    return schedules[:count]


def enumerate_schedules_fixtb(rank_dag: InstructionDAG, random_seed=0, count=100, maxchannels=32, maxtb=80):
    instrs = topo_sort_instrs(rank_dag)

    schedules: list[schedule] = []

    #                             <------------------hehehehe----------------------->
    for _, channel_assignment in zip(range(count), manual_channel_enumeration(instrs, maxchannels=maxchannels, seed=random_seed)):
        sched = assign_tbs(rank_dag, instrs, channel_assignment, tb_limit=maxtb, single=True)
        assert len(sched) == 1
        schedules.append(sched[0])
        print(f'{len(schedules)}/{count}')

    return schedules


def nondeterministic_channel_assignment(instrs: List[Op], rank_dag: InstructionDAG):

    rank_t = NewType('rank_t', int)
    chan_t = NewType('chan_t', int)
    instr_t = NewType('instr_t', int)
    flow_t = set[tuple[rank_t, rank_t]]

    def create_flow(f: List[rank_t]) -> flow_t:
        return set(zip(f[:-1], f[1:]))

    all_channels = lambda: set(map(chan_t, range(32)))


    @dataclass
    class state:
        availability: dict[tuple[rank_t, rank_t], set[chan_t]]
        flows: list[flow_t]
        flow_channels: list[chan_t]
        assignment: dict[instr_t, chan_t]

        def matching_flow(self, flow: flow_t) -> chan_t:
            if flow in self.flows:
                return self.flow_channels[self.flows.index(flow)]
            return chan_t(-1)


        def reserve(self, sender: rank_t, receiver: rank_t, channel: chan_t):
            if channel in self.availability[sender, receiver]:
                self.availability[sender, receiver].remove(channel)





    def dfs(op: instr_t, channels: set[chan_t], f: List[rank_t], st: state) -> list[state]:
        if instrs[op].is_local():
            st.assignment[op] = chan_t(0)
            return [st]
        elif instrs[op].is_send():
            match: instr_t = instr_t(instrs.index(instrs[op].recv_match)) # type: ignore
            sender: rank_t = rank_t(instrs[op].rank)
            receiver: rank_t = rank_t(instrs[match].rank)

            channels = channels.intersection(st.availability[sender, receiver])
            f.append(sender)

            if not instrs[match].is_fused():
                f.append(receiver)
                flow = create_flow(f)

                if instrs[op].channel != -1:
                    chs = [chan_t(instrs[op].channel)]
                    new_states = [st]
                else:
                    ch = st.matching_flow(flow)
                    if ch == -1:
                        # NOW WE NEED NONDETERMINISM!!!!!
                        new_states = []
                        chs = list(channels)
                        for ch in chs:
                            new_st = deepcopy(st)
                            new_st.flows.append(flow)
                            new_st.flow_channels.append(ch)
                            new_states.append(new_st)
                        
                    else:
                        chs = [ch]
                        new_states = [st]
                
                for ch, new_st in zip(chs, new_states):
                    new_st.assignment[op] = ch
                    new_st.assignment[match] = ch
                    new_st.reserve(sender, receiver, ch)
            else:
                new_states = dfs(match, channels, f, st)

                for new_st in new_states:
                    ch = new_st.assignment[match]
                    new_st.assignment[op] = ch
                    new_st.reserve(sender, receiver, ch)

            # print(f'DFS returning with {len(new_states)} possible states')
            return new_states

        return [st]


    states = [state(availability=defaultdict(all_channels), flows=[], flow_channels=[], assignment={})]

    # assign channels to flows
    for i, op in enumerate(instrs):
        if op.inst == Instruction.send and op.recv_match.is_fused(): # type: ignore
            print(f'calling dfs (instr {i+1}/{len(instrs)})')
            new_states = []
            for st in states:
                new_states += dfs(instr_t(i), all_channels(), [], st)

            states = new_states

    return states

                    
                


            


def channel_assignment(instrs, rank_dag):

    instrs = deepcopy(instrs)
    rank_dag = deepcopy(rank_dag)

    def all_channels():
        return set([x for x in range(32)])    # First handle flows - if an instruction at Rx is fused Rw->Rx->Ry and takes c
    # Then flow Rw->Rx->Rz must be ib a different channel c' where c!=c'
    rank2sendch = [defaultdict(all_channels) for _ in range(rank_dag.num_ranks)]
    rank2recvch = [defaultdict(all_channels) for _ in range(rank_dag.num_ranks)]

    # DFS through the InstructionDAG identifying flows
    def valid_send_ch(sender, receiver, ch):
        return ch in rank2sendch[sender][receiver]
    def valid_recv_ch(sender, receiver, ch):
        return ch in rank2recvch[receiver][sender]

    # Returns a channel this flow can be scheduled on, else -1 
    def is_matching_flow(flow):
        if flow in flows:
            return flow_channels[flows.index(flow)]
        return -1

    def reserve_channel(sender, receiver, ch):
        if ch in rank2sendch[sender][receiver]:
            rank2sendch[sender][receiver].remove(ch)
        if ch in rank2recvch[receiver][sender]:
            rank2recvch[receiver][sender].remove(ch)
    flows = []
    flow_channels = []

    def create_flow(f):
        flow = set()
        for i in range(1, len(f)):
            flow.add((f[i-1], f[i]))
        return flow
    
    def dfs(op: Op, channels: List[chan_t], f: List[int]):
        if op.is_local():
            op.channel = chan_t(0)
        elif op.is_send():
            match: Op = op.recv_match # type: ignore
            sender = op.rank
            receiver = match.rank
            # Available channels
            channels = rank2sendch[sender][receiver].intersection(rank2recvch[receiver][sender]).intersection(channels)
            f.append(op.rank)
            # If not a fused op use the first possible channel (send, recv/rrc)
            if not match.is_fused():
                f.append(match.rank)
                flow = create_flow(f)
                # If the user has already manually scheduled this onto a channel, respect it
                if op.channel != -1:
                    ch = op.channel
                else:
                    ch = is_matching_flow(flow)
                    if ch == -1: # No flow matched - use the smallest available channel
                        ch = min(channels)
                        flows.append(flow)
                        flow_channels.append(ch)

                op.channel = ch
                match.channel = ch
                reserve_channel(sender, receiver, ch)
            else:
                dfs(match, channels, f)
                ch = match.channel
                op.channel = ch
                reserve_channel(sender, receiver, ch)

    # Assign channels to flows
    for op in instrs:
        if op.inst == Instruction.send and op.recv_match.is_fused():
            dfs(op, all_channels(), [])

    # print(f'{len(instrs)} to end')
    # return {old_op: op.channel for old_op, op in zip(old_instrs, instrs)}
    return [op.channel for op in instrs]
