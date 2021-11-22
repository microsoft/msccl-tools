# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from typing import Callable, List

import sccl.language
import sccl.topologies
from sccl.language.collectives import AllToAll


def alltoall_mesh(nnodes: int, ngpus: int, nchannels: int, threadblocks: int) -> None:
    """Generate XML for 4-phase mesh alltoall algorithm.

    Args:
        nnodes (int): Number of nodes.
        ngpus (int): Number of GPUs per node.
        nchannels (int): Number of channels/instances.
    """
    nranks: int = nnodes * ngpus
    node_rank: Callable[[int], int] = lambda r: r // ngpus
    local_rank: Callable[[int], int] = lambda r: r % ngpus
    stride_idx: Callable[[int, int, int], int] = lambda r, step, n: n // step * (r % step) + r // step
    shift_channel: Callable[[int, int], int] = lambda chunk_idx, ch: chunk_idx + nranks * ch

    topology = sccl.topologies.fully_connected(nranks)
    collective = AllToAll(nranks, nchannels, inplace=False, name='alltoall')
    with sccl.language.SCCLProgram('alltoall_mesh', topology, collective, instances=1, protocol='Simple', threadblocks=threadblocks):
        # get device on all ranks
        devices: List[sccl.language.Process] = list(map(lambda r: sccl.language.Rank(r), range(nranks)))

        # create buffer
        for r in range(nranks):
            for p in range(4):
                devices[r].create_scratch(f'phase-{p}')

        for ch in range(nchannels):
            # phase-0: per-gpu (step=ngpus) stride copy
            for r in range(nranks):
                for peer in range(nranks):
                    chunk = devices[r].input(peer * nchannels + ch)
                    chunk.send(r, buffer='phase-0', index=shift_channel(stride_idx(peer, ngpus, nranks), ch), ch=ch)

            # phase-1: intra-node alltoall
            for r in range(nranks):
                for g in range(ngpus):
                    peer = g + node_rank(r) * ngpus
                    chunk = devices[r].scratch('phase-0', shift_channel(g * nnodes, ch), size=nnodes)
                    chunk.send(peer, buffer='phase-1', index=shift_channel(local_rank(r) * nnodes, ch), ch=ch)

            # phase-2: per-gpu (step=nnodes) stride copy
            for r in range(nranks):
                for peer in range(nranks):
                    chunk = devices[r].scratch('phase-1', shift_channel(peer, ch))
                    chunk.send(r, buffer='phase-2', index=shift_channel(stride_idx(peer, nnodes, nranks), ch), ch=ch)

            # phase-3: inter-node alltoall
            for r in range(nranks):
                for n in range(nnodes):
                    peer = local_rank(r) + n * ngpus
                    chunk = devices[r].scratch('phase-2', shift_channel(n * ngpus, ch), size=ngpus)
                    if nchannels > 1:
                        chunk.send(peer, buffer='phase-3', index=shift_channel(node_rank(r) * ngpus, ch), ch=ch)
                    else:
                        chunk.send(
                            peer,
                            buffer=sccl.language.Buffer.output,
                            index=shift_channel(node_rank(r) * ngpus, ch),
                            ch=ch
                        )

            # re-order chunks in channels
            if nchannels <= 1:
                continue
            for r in range(nranks):
                for peer in range(nranks):
                    chunk = devices[r].scratch('phase-3', shift_channel(peer, ch))
                    chunk.send(
                        r,
                        buffer=sccl.language.Buffer.output,
                        index=stride_idx(peer, nranks, nranks * nchannels) + ch,
                        ch=ch
                    )

        sccl.language.XML()
        sccl.language.Check()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--num_nodes',
        type=int,
        default=2,
        help='number of nodes',
    )
    parser.add_argument(
        '-g',
        '--gpus_per_node',
        type=int,
        default=4,
        help='gpus per node',
    )
    parser.add_argument(
        '-c',
        '--channels',
        type=int,
        default=2,
        help='number of channels',
    )

    parser.add_argument(
        '-t',
        '--threadblocks',
        type=int,
        default=0,
        help='number of threadblocks. Default: 0, SCCLang controlled',
    )
    args = parser.parse_args()

    alltoall_mesh(args.num_nodes, args.gpus_per_node, args.channels, args.threadblocks)