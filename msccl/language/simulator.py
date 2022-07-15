from __future__ import annotations

from heapq import heappop, heappush
import itertools
from sys import stderr
import inspect
from typing import Optional, Type, Generator

from msccl.language.ir import *

import networkx as nx # type: ignore

# TODO: PollEvent should know whether it will advance the ip or not, instead of threadblocks advancing their ip manually

dgx2_top = nx.Graph()
nodes = range(8)
switches = [f's{i}' for i in range(12)]
dgx2_top.add_nodes_from(nodes, switch=False)
dgx2_top.add_nodes_from(switches, switch=True)
dgx2_top.add_edges_from(itertools.product(nodes[:4], switches[:6]))
dgx2_top.add_edges_from(itertools.product(nodes[4:], switches[6:]))
dgx2_top.add_edges_from(zip(switches[:6], switches[6:]))

link_t = tuple[int, ...]

def get_links(topology: nx.Graph) -> dict[tuple[rank_t, rank_t], list[link_t]]:
    gpus: set[rank_t] = {n for n, d in topology.nodes(data=True) if not d['switch']}
    # assert nx.bipartite.is_bipartite_node_set(topology, gpus)

    def get_gpu_links(n1: rank_t, n2: rank_t) -> Generator[link_t, None, None]:
        allowed = nx.subgraph_view(topology, filter_node=lambda n: topology.nodes[n]['switch'] or n in (n1, n2))
        yield from map(tuple, nx.all_simple_paths(allowed, n1, n2))

    links = {}
    for n1, n2 in itertools.combinations(gpus, 2):
        links[n1, n2] = list(get_gpu_links(n1, n2))

    return links


EPS = 1e-10 # some events have to be ordered, we use EPS to ensure that
pipeline_latency: int = int(1e100)
send_buffering_threshold: int = 4 << 20

event_counter = itertools.count()

class Logger:
    def __init__(self, parent, fmt):
        self.parent = parent
        self.fmt = fmt

    @staticmethod
    def __show_prefix(pfx: str):
        return lambda self, msg: print(f'[{pfx}] {self.fmt.format(**inspect.currentframe().f_back.f_locals)}: {msg}') # type: ignore

    debug = __show_prefix('DEBUG') # 
    debug = lambda *args: None
    info = __show_prefix('INFO')
    warn = __show_prefix('WARN')
    error = __show_prefix('ERROR')


@dataclass 
class Msg:
    rank: rank_t
    chan: chan_t


@dataclass
class ChunkAvailMsg(Msg):
    def __hash__(self):
        return hash(('chunk', self.rank, self.chan))

    def __repr__(self) -> str:
        return f'(chunk {self.rank} {self.chan})'


@dataclass
class ConnAvailMsg(Msg):
    rank: rank_t
    chan: chan_t

    def __hash__(self):
        return hash(('conn', self.rank, self.chan))

    def __repr__(self) -> str:
        return f'(conn {self.rank} {self.chan})'

def event_priority(e: Type[Event]) -> int:
    if e is SubscribeEvent:
        return 0
    if e is PollEvent:
        return 2
    return 1

@dataclass
class Event:
    timestamp: float

    def __lt__(self, other: Event):
        return (self.timestamp, event_priority(type(self)), self.id) < (other.timestamp, event_priority(type(other)), other.id)

    def __post_init__(self, *args, **kwargs):
        self.id = next(event_counter)


@dataclass
class SubscribeEvent(Event):
    """Subscribe a threadblock to a message"""
    tb: TB # threadblock to subscribe
    msg: Msg # message to subscribe to

    def __repr__(self) -> str:
        return f'SubscribeEvent(tb={str(self.tb)}, msg={self.msg})'


@dataclass
class NotifyEvent(Event):
    """Notify all subscribers to a particular message"""
    msg: Msg

    def __repr__(self) -> str:
        return f'NotifyEvent(msg={self.msg})'


@dataclass
class PollEvent(Event):
    """Poll a threadblock for its next instruction to execute"""
    tb: TB

    def __repr__(self) -> str:
        return f'PollEvent(tb={str(self.tb)})'


@dataclass
class ExecEvent(Event):
    """Fired when an instruction is *finished* executing"""
    op: Op
    tb: TB

    def __repr__(self) -> str:
        return f'ExecEvent(tb={str(self.tb)}, op={self.op.inst})'




@dataclass
class AcquireChannel(Event):
    """Fires when a send has *just started*"""
    link: link_t | None
    rank: rank_t
    chan: chan_t

    def __repr__(self) -> str:
        return f'AcquireChannel(rank={self.rank}, chan={self.chan})'


@dataclass
class TryAcquire(Event):
    tb: TB
    conn: connection_t
    callbacks: list[Event]
    override_dirty: bool = False
    override_in_use: bool = False


@dataclass
class TryRecv(Event):
    tb: TB
    chan: chan_t
    rank: rank_t
    callbacks: list[Event]
    # callbacks_1: list[Event] # callbacks to execute if the receive is successful but there's more data
    # callbacks_2: list[Event] # callbacks to execute if the receive is successful and there's no more data



@dataclass
class SendAvailable(Event):
    """Fires when the first pipelined slice has finished sending"""
    rank: rank_t
    chan: chan_t

    def __repr__(self) -> str:
        return f'SendAvailable(rank={self.rank}, chan={self.chan})'


chunk = NewType('chunk', bool)

class buffer:
    def __init__(self, rank: rank_t, flag: Type[Msg]):
        self.chunks: Dict[chan_t, chunk] = defaultdict(lambda: chunk(False))
        self.rank = rank
        self.flag = flag
        self.world: World = None # type: ignore

    def set_world(self, world: World):
        self.world = world

    def put(self, chan: chan_t, val: bool, timestamp: float):
        self.chunks[chan] = chunk(val)
        # print(f'sending notify {self.flag(rank=self.rank, chan=chan)}')
        self.world.notify(self.flag(rank=self.rank, chan=chan), timestamp)

    def __call__(self, chan: chan_t):
        return self.chunks[chan]

def pretty_print(op: Op) -> str:
    return f'{op.channel}@{op.inst}:{op.src.rank, op.src.index}->{op.dst.rank, op.dst.index}'

class TB:
    def __init__(self, ops: list[Op]) -> None:
        self.ops = ops
        self.rank: Rank = None # type: ignore
        self.world: World = None # type: ignore

        self.ip: int = 0
        self.tbid: tbid_t = tbid_t(-1)

        self.log = Logger(self, '<TB {self.rank.id}:{self.tbid}|{timestamp}>')
    

    def set_rank(self, rank: Rank, tbid: tbid_t, world: World):
        self.rank = rank
        self.tbid = tbid
        self.world = world

        timestamp = -1

        self.log.debug(' ; '.join(map(pretty_print, self.ops)))


    def get_events(self, timestamp: float):
        if self.ip >= len(self.ops):
            return

        op = self.ops[self.ip]
        # self.log.debug(f'Polled, current ip is {self.ip} and op is {pretty_print(op)}')

        if op.is_recv():

            callbacks: list[Event] = []
            callbacks.append(ExecEvent(timestamp=timestamp + self.world.latency(op), op=op, tb=self))

            # callbacks_1: list[Event] = []
            # callbacks_1.append(ExecEvent(timestamp=timestamp + self.world.latency(op), op=op, tb=self))
            # callbacks_1.append(SubscribeEvent(
            #             timestamp=timestamp + self.world.latency(op), 
            #             tb=self, msg=ConnAvailMsg(self.rank.id, op.channel)))

            # callbacks_2: list[Event] = []
            # callbacks_2.append(ExecEvent(timestamp=timestamp + self.world.latency(op), op=op, tb=self))
            # callbacks_2.append(PollEvent(timestamp=timestamp + self.world.latency(op), tb=self))

            self.log.debug(f'Trying to receive on {self.rank.id, op.channel}')
            self.world.schedule(TryRecv(timestamp=timestamp, tb=self, chan=op.channel, rank=self.rank.id, callbacks=callbacks)) #callbacks_1=callbacks_1, callbacks_2=callbacks_2))

            # # check if data is available in the buffer
            # if self.rank.dirty(op.channel):
            #     # if it is, complete the receive (this will clear the dirty bit)
            #     self.log.debug(f'Initiated recv from {self.rank.id, op.channel}')
            #     self.world.schedule(ExecEvent(timestamp=timestamp + self.world.latency(op), op=op, tb=self))
            #     self.ip += 1 # don't check the recv next time we look at this thread

            #     # is the send still in-flight (i.e. there is more data to read)
            #     # (this happens when sends are pipelined)
            #     if self.rank.locked(op.channel):
            #         self.log.debug('Send not finished, subscribing to buffer')
            #         # this will notify the thread as soon as that send completes
            #         # a benefit of simulating everything is we don't actually have to process the receive of each slice :)
            #         self.world.schedule(SubscribeEvent(
            #             timestamp=timestamp + self.world.latency(op), 
            #             tb=self, msg=ConnAvailMsg(self.rank.id, op.channel)))
            #     else:
            #         # we read everything we needed to
            #         self.world.schedule(PollEvent(timestamp=timestamp + self.world.latency(op), tb=self))

            # else:
            #     self.log.debug(f'Trying to receive from {self.rank.id, op.channel}, waiting for data')
            #     # notify when the send completes (thus making the connection available)
            #     self.world.schedule(SubscribeEvent(
            #         timestamp=timestamp, tb=self, msg=ChunkAvailMsg(self.rank.id, op.channel)))

        elif op.is_send():

            callbacks = []
            callbacks.append(SendAvailable(
                    timestamp=timestamp + min(self.world.pipeline, self.world.latency(op)),
                    rank=op.dst.rank, chan=op.channel))
            callbacks.append(ExecEvent(timestamp=timestamp + self.world.latency(op), op=op, tb=self))
            callbacks.append(PollEvent(timestamp=timestamp + self.world.latency(op), tb=self))

            self.log.debug(f'Trying to acquire connection {op.src.rank, op.dst.rank, op.channel}')

            self.world.schedule(TryAcquire(
                timestamp=timestamp, tb=self, conn=(op.src.rank, op.dst.rank, op.channel), 
                callbacks=callbacks, override_dirty=(op.is_recv() and op.src.rank == op.dst.rank),
                override_in_use=(op.cnt() * self.world.chunksize) > send_buffering_threshold))

            # # is another send is currently using this channel?
            # if self.world.ranks[op.dst.rank].locked(op.channel):
            #     self.log.debug(f'Send in progress on chunk {op.dst.rank, op.channel}; subscribing to wait')
            #     self.world.schedule(SubscribeEvent(
            #         timestamp=timestamp, tb=self, msg=ConnAvailMsg(op.dst.rank, op.channel)))
            
            # # is there unread data in the buffer? (Note: this might be a fused op, in which case we're the ones meant to read that unread data as well) 
            # elif self.world.ranks[op.dst.rank].dirty(op.channel) and not (op.is_recv() and op.src.rank == op.dst.rank):
            #     self.log.debug(f'Unread send in buffer {op.dst.rank, op.channel}; subscribing to wait')
            #     self.world.schedule(SubscribeEvent(
            #         timestamp=timestamp, tb=self, msg=ChunkAvailMsg(op.dst.rank, op.channel)))

            # elif ((op.cnt() * self.world.chunksize > send_buffering_threshold) or True) and self.world.in_use[(link := self.world.mapping[op.src.rank, op.dst.rank, op.channel])]:
            #     self.log.debug('Link already in use, subscribing to wait') # TODO: maybe decide to do something more clever than this
            #     # raise SystemExit() # TODO: need to track the particular channel using the link, not just whether its in use or not
            # else: # no reason to wait; full steam ahead!
            #     self.log.debug(f'Initiated send to {op.dst.rank, op.channel} on link {link}')
                
            #     # acquire the channel immediately
            #     self.world.schedule(AcquireChannel(timestamp=timestamp, link=link, rank=op.dst.rank, chan=op.channel))

            #     # after the pipeline delay, mark the data as available
            #     self.world.schedule(SendAvailable(
            #         timestamp=timestamp + min(self.world.pipeline, self.world.latency(op)),
            #         rank=op.dst.rank, chan=op.channel))

            #     # once the send has finished, release the channel by firing Exec(send)
            #     self.world.schedule(ExecEvent(timestamp=timestamp + self.world.latency(op), op=op, tb=self))

            #     # keep running this thread
            #     self.world.schedule(PollEvent(timestamp=timestamp + self.world.latency(op), tb=self))
            #     self.ip += 1

        else:
            # self.log.debug(f'Scheduling local {op}')
            self.world.schedule(ExecEvent(timestamp=timestamp + self.world.latency(op), op=op, tb=self))
            self.world.schedule(PollEvent(timestamp=timestamp + self.world.latency(op), tb=self))
            self.ip += 1




    def __repr__(self):
        return f'TB {self.rank.id}:{self.tbid}'


class Rank:
    def __init__(self, rid: rank_t, tbs: list[TB]):
        self.tbs = tbs

        self.dirty = buffer(rid, ChunkAvailMsg) # chunks with unread data
        self.locked = buffer(rid, ConnAvailMsg) # chunks actively being written to
        self.active_read: dict[chan_t, bool] = defaultdict(bool) # chunks to which a read is in progress (cannot start a new read)
            
        self.id = rid
        self.world: World = None # type: ignore

    def set_world(self, world: World):
        for i, tb in enumerate(self.tbs):
            tb.set_rank(self, tbid_t(i), world)

        self.dirty.set_world(world)
        self.locked.set_world(world)

        self.world = world


connection_t = tuple[rank_t, rank_t, chan_t]

class World:

    top: nx.Graph = None
    links: dict[tuple[rank_t, rank_t], list[link_t]] = {}

    @classmethod
    def set_top(cls, top: nx.Graph):
        World.top = top
        World.links = get_links(top)

    def __init__(self, ranks: dict[rank_t, Rank], 
                        chunksize=1, 
                        timing_info=None, 
                        verbose=False,
                        num_conns=0,
                        mapping: dict[connection_t, link_t]={},
                        restrict_tbs: List[Tuple[rank_t, tbid_t]]=[], **kwargs):

        
        self.ranks = ranks
        self.queue: list[Event] = []
        self.trace: list[tuple[float, Op]] = []

        self.chunksize = chunksize
        self.timing_info = timing_info

        self.verbose = verbose
        self.log = Logger(self, '<world|{event.timestamp}>')
        

        for _, rank in ranks.items():
            rank.set_world(self)

        self.timestamp = 0.
        self.pipeline = int(1e100)

        if timing_info:
            self.timestamp = timing_info[3][0] * num_conns + timing_info[3][1]
            self.pipeline = timing_info[4]
            self.send_buffering_threshold = timing_info[5]
            

        self.subscribers: dict[Msg, list[TB]] = defaultdict(list)

        self.mapping = mapping
        self.in_use: dict[link_t, Optional[connection_t]] = defaultdict(lambda: None)

    def notify(self, msg: Msg, timestamp: float):
        # print(f'Scheduling notification for {self.subscribers[msg]}')
        self.schedule(NotifyEvent(timestamp=timestamp, msg=msg))
        # print(self.queue)


    def schedule(self, event: Event):
        heappush(self.queue, event)

    def initialize(self):
        for _, rank in self.ranks.items():
            for tb in rank.tbs:
                self.queue.append(PollEvent(0, tb))

    def debug_mode(self):
        print('Simulator paused; entering debug mode. Type "res" to resume or "quit" to quit')
        while True:
            line = input('debug> ')
            if line == 'res':
                break
            try:
                print(f'=> {eval(line)}')
            except Exception as e:
                print(e)
                
    def is_acquireable(self, conn: connection_t, override_dirty=False):
        _, dst, chan = conn
        # print(f'\tTry acquire for {conn}: ', end='')
        if self.ranks[dst].locked(chan):
            # print('FAIL: another send in progress to dest')
            return ConnAvailMsg(dst, chan)

        if self.ranks[dst].dirty(chan) and not override_dirty:
            # print('FAIL: unread data in buffer')
            return ChunkAvailMsg(dst, chan)

        link = self.mapping[conn]
        # print(f'[link={link},conn={self.in_use[link]}]')

        if (blocker := self.in_use[link]):
            # print(f'FAIL: link {link} in use by {blocker}')
            return ConnAvailMsg(blocker[1], blocker[2])
        # print('SUCCESS')
        return None

    def acquire(self, timestamp: float, conn: connection_t, mark_in_use=True):
        _, dst, chan = conn
        # print(f'locking (conn {dst} {chan})')
        self.ranks[dst].locked.put(chan, True, timestamp)
        # print(f'Acquiring link {self.mapping[conn]} for {conn}')
        if mark_in_use:
            self.in_use[self.mapping[conn]] = conn
        # else:
        #     print('Skipping acquire')

    def release(self, timestamp: float, conn: connection_t):
        _, dst, chan = conn
        # print(f'unlocking (conn {dst} {chan})')
        self.ranks[dst].locked.put(chan, False, timestamp)
        # print(f'Releasing link {self.mapping[conn]}')
        self.in_use[self.mapping[conn]] = None

    def recvable(self, rank: rank_t, channel: chan_t):
        return self.ranks[rank].dirty(channel) and not self.ranks[rank].active_read[channel]

    def recv_lock(self, rank: rank_t, channel: chan_t):
        self.ranks[rank].active_read[channel] = True

    def recv_unlock(self, timestamp: float, rank: rank_t, channel: chan_t):
        assert self.ranks[rank].dirty(channel)
        self.ranks[rank].dirty.put(channel, False, timestamp)
        self.ranks[rank].active_read[channel] = False


    def clear_tb_polls(self, tb: TB):
        self.queue = list(filter(lambda e: isinstance(e, PollEvent) and e.tb == tb, self.queue))

    def run(self) -> float:
        try:
            while len(self.queue):
                # print(f'Subscribers: {self.subscribers}')
                event = heappop(self.queue)
                # self.log.info(set(filter(lambda k: self.in_use.get, self.in_use.keys())))
                if isinstance(event, PollEvent):
                    event.tb.get_events(event.timestamp)
                elif isinstance(event, ExecEvent):
                    self.execute(event)
                elif isinstance(event, SubscribeEvent):
                    if event.tb not in self.subscribers[event.msg]:
                        self.log.debug(f'Subscribing {str(event.tb)} to {event.msg}')
                        self.subscribers[event.msg].append(event.tb)
                    else:
                        self.log.debug(f'{str(event.tb)} is already subscribed to {event.msg}; skipping')
                    # print(self.subscribers)
                elif isinstance(event, NotifyEvent):
                    # print(self.subscribers)
                    for sub in self.subscribers[event.msg]:
                        self.log.debug(f'Notifying {str(sub)} of {event.msg}')
                        heappush(self.queue, PollEvent(timestamp=event.timestamp, tb=sub))
                    self.subscribers[event.msg] = []
                    # print(self.queue)
                elif isinstance(event, AcquireChannel):
                    self.log.debug(f'Channel {event.rank, event.chan} locked')
                    self.ranks[event.rank].locked.put(event.chan, True, event.timestamp)

                    # mark the link as in use
                    if event.link is not None:
                        self.log.warn('USING DEPRECATED AcquireChannel API!!!!!')
                        self.in_use[event.link] = (event.rank, event.rank, event.chan)

                elif isinstance(event, TryAcquire):
                    if not (blocker := self.is_acquireable(event.conn, event.override_dirty)):
                        self.acquire(event.timestamp, event.conn, mark_in_use=event.override_in_use)
                        self.log.debug(f'Acquire succeeded: callbacks = {event.callbacks}')
                        for callback in event.callbacks:
                            self.schedule(callback)

                        event.tb.ip += 1
                    else:
                        _, dst, chan = event.conn
                        # self.log.debug(f'Acquire failed: {blocker}')
                        # if blocker == 'conn':
                        #     msg: Msg = ConnAvailMsg(dst, chan)
                        # elif blocker == 'chunk':
                        #     msg = ChunkAvailMsg(dst, chan)
                        self.schedule(SubscribeEvent(
                            timestamp=event.timestamp, tb=event.tb, msg=blocker))

                elif isinstance(event, TryRecv):
                    if self.recvable(event.rank, event.chan):
                        self.recv_lock(event.rank, event.chan)
                        self.log.debug(f'Recv started, callbacks = {event.callbacks}')
                        for callback in event.callbacks:
                            self.schedule(callback)
                        event.tb.ip += 1
                    else:
                        self.log.debug('Recv failed')
                        self.schedule(SubscribeEvent(
                            timestamp=event.timestamp, tb=event.tb, msg=ChunkAvailMsg(event.rank, event.chan)))

                    # if evt_rank.dirty(event.chan):
                    #     if evt_rank.locked(event.chan):
                    #         msg = 'more data pending'
                    #         callbacks = event.callbacks_1
                    #     else:
                    #         msg = 'no more data'
                    #         callbacks = event.callbacks_2
                    #     self.log.debug(f'Recv succeeded, {msg}: callbacks = {callbacks}')
                    #     for callback in callbacks:
                    #         self.schedule(callback)
                    #     event.tb.ip += 1
                    # else:
                    #     self.log.debug('Recv failed, no data available')
                    #     self.schedule(SubscribeEvent(
                    #         timestamp=event.timestamp, tb=event.tb, msg=ChunkAvailMsg(event.rank, event.chan)))
                elif isinstance(event, SendAvailable):
                    self.log.debug(f'Data available from send on (rank, channel) {event.rank, event.chan}')
                    self.ranks[event.rank].dirty.put(event.chan, True, event.timestamp)

        except KeyboardInterrupt:
            self.debug_mode()

        try:
            for rank in self.ranks:
                for tb in self.ranks[rank].tbs:
                    assert tb.ip == len(tb.ops), f'{str(tb)} did not complete!'
        except AssertionError as e:
            self.log.error(e)
            quit()
            # self.debug_mode()
        return self.timestamp

    def execute(self, event: ExecEvent):

        # all fused ops follow the pattern `[recv]?[local]*[send]?`
        if event.op.is_recv():
            # the action of a receive is to clear the dirty bit, marking the chunk as being read
            self.log.debug(f'From {str(event.tb)}: Executing recv to {event.op.rank, event.op.channel}')
            self.recv_unlock(event.timestamp, event.op.rank, event.op.channel)

            # manually schedule the next callback here
            if self.ranks[event.op.rank].locked(event.op.channel):
                self.log.debug('More data pending')
                self.schedule(SubscribeEvent(timestamp=event.timestamp, tb=event.tb, msg=ConnAvailMsg(rank=event.op.rank, chan=event.op.channel)))
            else:
                self.log.debug('All data read')
                self.schedule(PollEvent(timestamp=event.timestamp, tb=event.tb))
            # if self.ranks[event.op.rank].dirty(event.op.channel):
            #     self.ranks[event.op.rank].dirty.put(event.op.channel, False, event.timestamp)
            # else:
            #     # # somebody else read the chunk between issuing and executing this receive
            #     # # subscribe to watch the chunk again
            #     # msg = ChunkAvailMsg(event.op.rank, event.op.channel)
            #     # self.subscribers[msg].add(event.tb)
            #     # # and uhhh the threadblock is gonna have to actually go back and re-execute it
            #     # self.clear_tb_polls(event.tb)
            #     # raise SystemExit()
            #     self.schedule(PollEvent(timestamp=event.timestamp, tb=event.tb))
            #     event.tb.ip -= 1
        if event.op.is_local():
            # local ops don't really have to be simulated
            pass # self.log.debug(f'Executing local {event.op} for {str(event.tb)}')

        if event.op.is_send():
            self.log.debug(f'From {str(event.tb)}: Executing send to {event.op.dst.rank, event.op.channel}')
            self.release(event.timestamp, (event.op.src.rank, event.op.dst.rank, event.op.channel))
            # # the action of a send is to release the channel lock...
            
            # assert self.ranks[event.op.dst.rank].locked(event.op.channel)
            # self.ranks[event.op.dst.rank].locked.put(event.op.channel, False, event.timestamp)
            
            # # ...and also mark the link as free
            # self.in_use[(link := self.mapping[event.op.src.rank, event.op.dst.rank, event.op.channel])] = None
            # # self.log.info(f'Freed link {link}!')

        self.trace.append((event.timestamp, event.op))
        self.timestamp = event.timestamp
        if self.verbose:
            print(f'\t** TB {event.tb.rank.id}:{event.tb.tbid} executing {event.op.inst} @ t = {event.timestamp} **', file=stderr)


    def get_congestion(self):
        return 1 # congestion scales the bandwidth, lower = more congested



    def base_cost(self, inst: Instruction, count: int = 1):
        # really only supposed to work on chunk sizes up to 128 MB
        breaks: dict[Instruction, tuple[float, float]] = {
            Instruction.copy: (5390., 2337591.),
            Instruction.reduce: (8388353., 64833826),
            Instruction.send: (4954., 2138062.)
        }

        slopes: dict[Instruction, tuple[float, float, float]] = {
            Instruction.copy: (5.36313E-4, 3.00438E-5, 3.14277E-5),
            Instruction.reduce: (1.0578E-4, 1.0395E-4, 1.0622E-4),
            Instruction.send: (3.04591E-4, 3.07056E-5, 3.28184E-5)
        }
        # :sweat_smile:
        ntrcps: dict[Instruction, tuple[float, float, float]] = {
            Instruction.copy: (2.16458, 4.89355, 1.65851),
            Instruction.reduce: (1.35567, 16.63357, -130.21764),
            Instruction.send: (5.29086, 6.64778, 2.13039)
        }

        if self.timing_info is not None:
            breaks, slopes, ntrcps, _, _, _ = self.timing_info

        def get_idx(val, cutoffs):
            low, high = cutoffs
            if val < low:
                return 0
            elif val < high:
                return 1
            return 2

        def linterpolate(val, cutoffs, ms, bs):
            idx = get_idx(val, cutoffs)
            return val * ms[idx] + bs[idx]

        if inst is Instruction.recv:
            inst = Instruction.copy
        
        return linterpolate(self.chunksize * count, breaks[inst], slopes[inst], ntrcps[inst])
        
        
    def latency(self, op: Op):
        if op.inst in (Instruction.reduce, Instruction.recv, Instruction.copy, Instruction.send):
            return self.base_cost(op.inst, op.cnt())
        
        if op.inst is Instruction.recv_copy_send:
            return self.base_cost(Instruction.recv, op.cnt()) + self.base_cost(Instruction.copy, op.cnt()) + self.base_cost(Instruction.send, op.cnt())
        
        if op.inst is Instruction.recv_reduce_copy:
            return self.base_cost(Instruction.recv, op.cnt()) + self.base_cost(Instruction.reduce, op.cnt()) + self.base_cost(Instruction.copy, op.cnt())

        if op.inst is Instruction.recv_reduce_copy_send:
            return self.base_cost(Instruction.recv, op.cnt()) + self.base_cost(Instruction.reduce, op.cnt()) + self.base_cost(Instruction.copy, op.cnt()) + self.base_cost(Instruction.send, op.cnt())
        
        if op.inst is Instruction.recv_reduce_send:
            return self.base_cost(Instruction.recv, op.cnt()) + self.base_cost(Instruction.reduce, op.cnt()) + self.base_cost(Instruction.send, op.cnt())

        print(f'[WARN] Unrecognized opcode: {op.inst}; assuming 0 latency')
        return 0
            

def get_connections(prog: Program) -> set[connection_t]:
    conns: set[connection_t] = set()
    for gpu in prog.gpus:
        for tb in gpu.threadblocks:
            conns.add((tb.recv, gpu.rank, tb.channel))

    conns = {c for c in conns if -1 not in c}

    return conns


def schedule(connections: set[connection_t], links: dict[tuple[rank_t, rank_t], list[link_t]]) -> dict[connection_t, link_t]:
    channels: dict[tuple[rank_t, rank_t], list[chan_t]] = defaultdict(list)
    for r1, r2, c in connections:
        channels[r1, r2].append(c)

    mapping: dict[connection_t, tuple[int, ...]] = {}
    for r1, r2 in channels:
        for i, chan in enumerate(channels[r1, r2]):
            if (r1, r2) in links:
                avail = links[r1, r2]
            else:
                avail = links[r2, r1]
            mapping[r1, r2, chan] = tuple(avail[i % len(avail)])

    return mapping

    


def build_world(prog: Program, **params) -> World:
    ranks: dict[rank_t, Rank] = {}
    
    for gpu in prog.gpus:
        tbs: list[TB] = []
        for tb in gpu.threadblocks:
            for op in tb.ops:
                op.channel = tb.channel
            tbs.append(TB(ops=tb.ops))
            
        ranks[gpu.rank] = Rank(rid=gpu.rank, tbs=tbs)

    connections = get_connections(prog)
    # params['num_conns'] = len(gpu.threadblocks) # len({(r1, r2 )for (r1, r2, _) in connections})
    # params['mapping'] = schedule(connections, World.links)


    return World(ranks, num_cons=len(gpu.threadblocks), mapping=schedule(connections, World.links), **params)

    