# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.algorithm import *
from sccl.shortest_path_sets import *
from z3 import *

class UflraOptimizer(object):
    def __init__(self, topology, collective):
        self.topology = topology
        self.collective = collective
        self.spsets = shortest_path_sets(topology, collective)

    def optimize(self):
        opt = Optimize()

        C = DeclareSort('Chunk')
        R = DeclareSort('Ranks')
        send = Function('send', C, R, R, RealSort())
        start = Function('start', C, R, RealSort())

        time = Real('time')
        opt.minimize(time)

        def chunkc(c):
            return Const(f'chunk{c}', C)

        def rankc(r):
            return Const(f'rank{r}', R)

        def latency(src, dst):
            assert len(self.topology.switches) == 0
            return 1 / self.topology.link(src, dst)

        for chunk in self.collective.chunks():
            c = chunkc(chunk)
            for rank in self.collective.ranks():
                r = rankc(rank)
                if not rank in self.spsets[chunk]:
                    continue
                if self.collective.precondition(rank, chunk):
                    # Have chunks start on their starting ranks before the first step
                    # This is not required for the encoding, but makes debugging the models produced more intuitive
                    opt.add(start(c, r) == 0)
                else:
                    # Any rank that gets a chunk (and doesn't start with it) must have a source for it
                    opt.add(Implies(start(c, r) <= time,
                        Or([send(c, rankc(src), r) + latency(src, rank) <= start(c, r)
                            for src in self.topology.sources(rank) if src in self.spsets[chunk]])))
                # If the postcondition requires the chunk on the rank then it must start being there before the end
                if self.collective.postcondition(rank, chunk):
                    opt.add(start(c, r) <= time)
                for src in self.topology.sources(rank):
                    if not src in self.spsets[chunk]:
                        continue
                    s = rankc(src)
                    # Senders need to have the chunk at send time
                    opt.add(Implies(send(c, s, r) <= time, start(c, s) <= send(c, s, r)))
                    for other in range(chunk):
                        o = chunkc(other)
                        if not rank in self.spsets[other] or not src in self.spsets[other]:
                            continue
                        opt.add(Implies(And(send(c, s, r) <= time, send(o, s, r) <= time), Or(
                            send(c, s, r) + latency(src, rank) <= send(o, s, r),
                            send(o, s, r) + latency(src, rank) <= send(c, s, r))))

        print('Encoded', flush=True)
        opt.check()
        model = opt.model()

        sends = defaultdict(list)
        total_time = model.eval(time).as_fraction()
        for chunk in self.collective.chunks():
            for dst in self.collective.ranks():
                if not dst in self.spsets[chunk]:
                    continue
                for src in self.topology.sources(dst):
                    if not src in self.spsets[chunk]:
                        continue
                    send_time = model.eval(send(chunkc(chunk), rankc(src), rankc(dst))).as_fraction()
                    if send_time < total_time:
                        sends[send_time].append((chunk, src, dst))
        
        times = sorted(sends.keys())
        for time in times:
            print(f'Time {time}')
            for chunk, src, dst in sends[time]:
                print(f'  Send {chunk} on {src}→{dst}')
        print(f'Total time {total_time}')

        return sends
