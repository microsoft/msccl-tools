from __future__ import annotations
from heapq import heappop, heappush
from sys import stderr
import inspect
from typing import Any

from msccl.language.ir import *
from numpy import isin

# TODO: There is a bug with fused instructions
# TODO: e.g. rrcs where receive and send are the same peer
# TODO: dirty bit means send can't happen, so receive doesn't happen, so bit never gets cleared

EPS = 1e-10 # some events have to be ordered, we use EPS to ensure that

class Logger:
    def __init__(self, parent, fmt):
        self.parent = parent
        self.fmt = fmt

    @staticmethod
    def __show_prefix(pfx: str):
        return lambda self, msg: print(f'[{pfx}] {self.fmt.format(**inspect.currentframe().f_back.f_locals)}: {msg}') # type: ignore

    debug = __show_prefix('DEBUG') # 
    # debug = lambda *args: None
    info = __show_prefix('INFO')
    warn = __show_prefix('WARN')
    error = __show_prefix('ERROR')



# TODO: add global launch overhead, add pipelining to autotuner
# TODO: receive can't finish before send; take max of (pipeline + copy, send finish)
# TODO: cross-TB dependence
pipeline_latency: int = int(1e100)

@dataclass
class Event:
    timestamp: float

    def __lt__(self, other: Event):
        return self.timestamp < other.timestamp

@dataclass
class PollEvent(Event):
    tb: TB
    def __repr__(self) -> str:
        return f'PollEvent(tb={self.tb.rank.id}:{self.tb.tbid})'

@dataclass
class ExecEvent(Event):
    op: Op
    tb: TB

    def __repr__(self) -> str:
        return f'ExecEvent(tb={self.tb.rank.id}:{self.tb.tbid}, op={self.op.inst})'


@dataclass
class FinishSendEvent(Event):
    rank: rank_t
    chan: chan_t


@dataclass
class RecvEvent(Event):
    tb: TB
    chan: chan_t

    def __repr__(self) -> str:
        return f'RecvEvent(tb={self.tb.rank.id}:{self.tb.tbid}, chan={self.chan})'

@dataclass
class Flag:
    ...

@dataclass
class ChunkFlag(Flag):
    rank: rank_t
    chan: chan_t

    def __hash__(self):
        return hash(('chunk', self.rank, self.chan))


@dataclass
class AwaitSendFlag(Flag):
    rank: rank_t
    chan: chan_t

    def __hash__(self):
        return hash(('await', self.rank, self.chan))


@dataclass
class OpFlag(Flag):
    tbid: tbid_t
    step: int

    def __hash__(self):
        return hash(('op', self.tbid, self.step))

@dataclass
class SubscribeEvent(Event):
    tb: TB
    flag: Flag

    # rank: rank_t
    # chan: chan_t # = chan_t(-1)


@dataclass
class NotifyEvent(Event):
    flag: Flag



chunk = NewType('chunk', bool)
buffer = dict[chan_t, chunk]

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

    def get_events(self, timestamp: float) -> list[Event]:
        events: list[Event] = []

        if self.ip >= len(self.ops):
            # out of operations to run!
            return [] # the simulator won't even poll us again

        self.log.debug(f'Polled, current ip is {self.ip} and op is {pretty_print(self.ops[self.ip])}')

        op = self.ops[self.ip]

        if op.is_recv():
            # if data is available in the buffer
            if self.rank.remote_buffer[op.channel]:
                self.log.debug(f'Initiated receive from {self.rank.id, op.channel}')
                events.append(ExecEvent(timestamp=timestamp + self.world.latency(op), op=op, tb=self))
                events.append(NotifyEvent(timestamp=timestamp + self.world.latency(op) + EPS, flag=ChunkFlag(self.rank.id, op.channel))) # executing the receive cleared the chunk, so notify anybody who cares

                # the data we read is all of it, so we can continue
                if not self.rank.active_sends[op.channel]:
                    self.log.debug(f'Finished receive, continuing execution')
                    events.append(PollEvent(timestamp=timestamp + self.world.latency(op), tb=self))
                    self.ip += 1
                else:
                    self.log.debug(f'Send not finished, subscribing to buffer')
                    # we need to wait for the send to finish before we continue; subscribe to that location in the active_sends buffer
                    events.append(SubscribeEvent(timestamp=timestamp + self.world.latency(op), tb=self, flag=AwaitSendFlag(self.rank.id, op.channel)))
            else:
                self.log.debug(f'Trying to receive from {self.rank.id, op.channel}, waiting for data')
                events.append(SubscribeEvent(timestamp=timestamp, tb=self, flag=ChunkFlag(self.rank.id, op.channel)))

        elif op.is_send():
            # do we need to block?
            if self.world.ranks[op.dst.rank].active_sends[op.channel]: # if there's an instruction actively writing to that chunk, wait until its finished
                self.log.debug(f'Send in progress on chunk {op.dst.rank, op.channel}; subscribing to wait')
                events.append(SubscribeEvent(timestamp=timestamp, tb=self, flag=AwaitSendFlag(op.dst.rank, op.channel)))
            elif self.world.ranks[op.dst.rank].remote_buffer[op.channel] and not (op.is_recv() and op.src.rank == op.dst.rank): # if there's no active send but there's data in the buffer and we aren't the ones meant to read it, wait until its cleared
                self.log.debug(f'Unread send in buffer {op.dst.rank, op.channel}; subscribing to wait')
                events.append(SubscribeEvent(timestamp=timestamp, tb=self, flag=ChunkFlag(op.dst.rank, op.channel)))
            else: # nothing in the way, go ahead and send!!
                # this will mark the chunk as containing data, which should be available after the pipeline delay, if its shorter than the whole send
                self.log.debug(f'Initiated send to {op.dst.rank, op.channel}')
                events.append(ExecEvent(timestamp=timestamp + min(pipeline_latency, self.world.latency(op)), op=op, tb=self))

                # since it sets the chunk in the buffer, we need to notify anybody who's listening
                events.append(NotifyEvent(timestamp=timestamp + min(pipeline_latency, self.world.latency(op)) + EPS, flag=ChunkFlag(op.dst.rank, op.channel)))

                # also schedule an event to let us know the send is finished
                events.append(FinishSendEvent(timestamp=timestamp + self.world.latency(op) + EPS, rank=op.dst.rank, chan=op.channel))

                # this also changes the active_sends buffer, so we need to send another notification
                events.append(NotifyEvent(timestamp=timestamp + self.world.latency(op) + EPS + EPS, flag=AwaitSendFlag(op.dst.rank, op.channel)))

                # we've sent successfully, keep polling this thread
                events.append(PollEvent(timestamp=timestamp + self.world.latency(op), tb=self))

                self.ip += 1
        else:
            self.log.debug(f'Scheduling local {op}')
            # just a local instruction, go ahead and schedule it
            events.append(ExecEvent(timestamp=timestamp + self.world.latency(op), op=op, tb=self))
            # and keep polling this thread
            events.append(PollEvent(timestamp=timestamp + self.world.latency(op), tb=self))

            self.ip += 1

        return events
            # block_send: bool = self.world.ranks[op.dst.rank].active_sends[op.channel] or (self.world.ranks[op.dst.rank].remote_buffer[op.channel] and not (op.is_recv() and op.src.rank == op.dst.rank))




        if op.is_recv() and not self.rank.remote_buffer[op.channel]:
            # if no data is available in the chunk, subscribe to be notified when data becomes available
            self.log.debug(f'Tried to receive {self.rank.id, op.channel} but not ready yet; emitting subcribe')
            events.append(SubscribeEvent(timestamp=timestamp, tb=self, flag=ChunkFlag(self.rank.id, op.channel)))
        elif op.is_recv() and self.rank.active_sends[op.channel]:
            # if data is available but a send is still active, 
            self.log.debug(f'Starting receive on {self.rank.id, op.channel} but cannot finish yet as send is still in progress')
            events.append(ExecEvent(timestamp=timestamp + self.world.latency(op), op=op, tb=self))
            events.append(SubscribeEvent(timestamp=timestamp + self.world.latency(op), tb=self, flag=AwaitSendFlag(self.rank.id, op.channel)))
        elif op.is_send():
            # if the bit is set, there's a pending send so we have to block until its resolved (either data in the buffer that hasn't been read, or an instruction is actively writing to this location)
            block_send: bool = self.world.ranks[op.dst.rank].remote_buffer[op.channel]# or self.world.ranks[op.dst.rank].active_sends[op.channel] 
            # HOWEVER, if that send is pending on a receive fused with this instruction, allow it :)
            if block_send and op.is_recv() and op.src.rank == op.dst.rank:
                block_send = False

            if block_send:
                self.log.debug(f'Pending send on {op.dst.rank, op.channel}, subscribing')
                events.append(SubscribeEvent(timestamp=timestamp, tb=self, flag=ChunkFlag(op.dst.rank, op.channel)))
            else:
                # make sure the chunk is ready after the pipelining delay so that any receives can read at the right time
                events.append(ExecEvent(timestamp=timestamp + min(pipeline_latency, self.world.latency(op)), op=op, tb=self))
                
                # # mark the chunk as fully written to
                events.append(FinishSendEvent(timestamp=timestamp + self.world.latency(op), rank=op.dst.rank, chan=op.channel))

                # continue polling this thread
                events.append(PollEvent(timestamp=timestamp + self.world.latency(op), tb=self))
                self.ip += 1
        else:
            # schedule the instruction now
            events.append(ExecEvent(timestamp=timestamp + self.world.latency(op), op=op, tb=self))
            
            # continue polling this thread
            events.append(PollEvent(timestamp=timestamp + self.world.latency(op), tb=self))
            self.ip += 1
        return events


class Rank:
    def __init__(self, rid: rank_t, tbs: list[TB]):
        self.tbs = tbs

        self.remote_buffer: buffer = defaultdict(lambda: chunk(False))
        self.active_sends: buffer = defaultdict(lambda: chunk(False))
            
        self.id = rid
        self.world: World = None # type: ignore

    def set_world(self, world: World):
        for i, tb in enumerate(self.tbs):
            tb.set_rank(self, tbid_t(i), world)
        self.world = world


class World:
    def __init__(self, ranks: dict[rank_t, Rank], alpha=3, beta=7, chunksize=1, timing_info=None, verbose=False, restrict_tbs: List[Tuple[rank_t, tbid_t]]=[]):
        
        self.ranks = ranks
        self.queue: list[Event] = []
        self.trace: list[tuple[float, Op]] = []
        

        self.alpha = alpha
        self.beta = beta
        self.chunksize = chunksize
        self.timing_info = timing_info

        self.verbose = verbose
        self.log = Logger(self, '<world|{event.timestamp}>')
        

        for _, rank in ranks.items():
            rank.set_world(self)

        self.timestamp = 0.

        self.subscribers: dict[Flag, set[TB]] = defaultdict(set)

        if len(restrict_tbs):
            for rank in self.ranks:
                for tbid in range(len(self.ranks[rank].tbs)):
                    if (rank, tbid_t(tbid)) not in restrict_tbs:
                        print(f'[WARN] Deleting ops for TB {rank}:{tbid}')
                        self.ranks[rank].tbs[tbid].ops = []

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
                

    def run(self) -> float:
        try:
            while len(self.queue):
                event = heappop(self.queue)
                # self.log.debug(f'Processing event type {type(event)}')
                if isinstance(event, PollEvent):
                    new_events = event.tb.get_events(event.timestamp)
                    for ev in new_events:
                        # self.log.info(f'Scheduling {ev} for {ev.timestamp}')
                        heappush(self.queue, ev)
                elif isinstance(event, ExecEvent):
                    self.execute(event)
                elif isinstance(event, RecvEvent):
                    event.tb.rank.remote_buffer[event.chan] = chunk(True)
                    if event.tb.ops[event.tb.ip].is_recv():
                        heappush(self.queue, PollEvent(timestamp=event.timestamp, tb=event.tb))
                elif isinstance(event, SubscribeEvent):
                    self.log.debug(f'Subscribing TB {event.tb.rank.id}:{event.tb.tbid} to {event.flag}')
                    self.subscribers[event.flag].add(event.tb)
                elif isinstance(event, NotifyEvent):
                    for sub in self.subscribers[event.flag]:
                        self.log.debug(f'Notifying TB {sub.rank.id}:{sub.tbid} of {event.flag}')
                        heappush(self.queue, PollEvent(timestamp=event.timestamp, tb=sub))
                    self.subscribers[event.flag] = set()
                elif isinstance(event, FinishSendEvent):
                    self.log.debug(f'Send finished on {event.rank, event.chan} at {event.timestamp}')
                    self.ranks[event.rank].active_sends[event.chan] = chunk(False)

                    # for subscriber in self.subscribers[AwaitSendFlag(event.rank, event.chan)]:
                    #     self.log.debug(f'Notifying {subscriber.rank.id}:{subscriber.tbid} of completed send')
                    #     heappush(self.queue, PollEvent(timestamp=event.timestamp, tb=subscriber))

        except KeyboardInterrupt:
            self.debug_mode()

        for rank in self.ranks:
            for tb in self.ranks[rank].tbs:
                assert tb.ip == len(tb.ops), f'TB {rank}:{tb.tbid} did not complete!'

        return self.timestamp

    def execute(self, event: ExecEvent):

        # def notify(tb: TB):
        #     heappush(self.queue, PollEvent(timestamp=event.timestamp, tb=tb))

        if event.op.is_send():
            self.log.debug(f'(From TB {event.tb.rank.id}:{event.tb.tbid}): Executing send to {event.op.dst.rank, event.op.channel}')
            self.ranks[event.op.dst.rank].remote_buffer[event.op.channel] = chunk(True)
            self.ranks[event.op.dst.rank].active_sends[event.op.channel] = chunk(True)

            # for subscriber in self.subscribers[ChunkFlag(event.op.dst.rank, event.op.channel)]:
            #     notify(subscriber)
            # self.log.debug(f'{len(self.subscribers[ChunkFlag(event.op.dst.rank, event.op.channel)])} subscribers notified!')
            # self.log.debug(f'(queue has {len(self.queue)} events)')

        elif event.op.is_recv():
            self.log.debug(f'(From TB {event.tb.rank.id}:{event.tb.tbid}): Executing receive to {event.op.rank, event.op.channel}')
            self.ranks[event.op.rank].remote_buffer[event.op.channel] = chunk(False)
            # for subscriber in self.subscribers[ChunkFlag(event.op.rank, event.op.channel)]:
            #     notify(subscriber)

        else:
            self.log.debug(f'Executing local {event.op} for TB {event.tb.rank.id}:{event.tb.tbid}')

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
            breaks, slopes, ntrcps = self.timing_info

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


    # def latency(self, op: Op):
    #     base_cost: Dict[Instruction, float] = {
    #         Instruction.reduce: 5,
    #         Instruction.copy: 1,
    #         Instruction.recv: 3
    #     }

    #     fuse_factor = 0.5
    #     chunksize = self.chunksize * op.cnt()

    #     match op.inst:
    #         case Instruction.reduce: 
    #             return base_cost[Instruction.reduce] * chunksize
    #         case Instruction.recv:
    #             return base_cost[Instruction.recv] * chunksize
    #         case Instruction.copy:
    #             return base_cost[Instruction.copy] * chunksize
    #         case Instruction.send:
    #             return self.alpha + chunksize * self.beta
    #         case Instruction.recv_copy_send:
    #             return (chunksize * (base_cost[Instruction.recv] + base_cost[Instruction.copy]) + self.alpha) * fuse_factor + chunksize * self.beta
    #         case Instruction.recv_reduce_copy:
    #             return (base_cost[Instruction.recv] + base_cost[Instruction.reduce] + base_cost[Instruction.copy]) * fuse_factor * chunksize
    #         case Instruction.recv_reduce_send:
    #             return (chunksize * (base_cost[Instruction.recv] + base_cost[Instruction.reduce]) + self.alpha) * fuse_factor + chunksize * self.beta
    #         case Instruction.recv_reduce_copy_send:
    #             return (chunksize * (base_cost[Instruction.recv] + base_cost[Instruction.reduce] + base_cost[Instruction.copy]) + self.alpha) * fuse_factor + chunksize * self.beta
    #         case Instruction.nop:
    #             return 0

    #     print(f'[WARN] Instruction {op.inst} not recognized; assuming 0 latency')
    #     return 0
            

        

def build_world(prog: Program, **params) -> World:
    ranks: dict[rank_t, Rank] = {}
    
    for gpu in prog.gpus:
        tbs: list[TB] = []
        for tb in gpu.threadblocks:
            for op in tb.ops:
                op.channel = tb.channel
            tbs.append(TB(ops=tb.ops))
            
        ranks[gpu.rank] = Rank(rid=gpu.rank, tbs=tbs)


    return World(ranks, **params)

    