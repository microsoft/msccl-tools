# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.collectives import allreduce
from msccl.algorithm import *
from msccl.instance import *

def compose_allreduce(reducescatter_algo, allgather_algo, logging=False):
    if reducescatter_algo.is_pipelined() or allgather_algo.is_pipelined():
        raise ValueError('Pipelining is not supported.')

    if reducescatter_algo.instance.chunks != allgather_algo.instance.chunks:
        raise ValueError(f'ReduceScatter and Allgather must have the same chunks (got {reducescatter_algo.instance.chunks} and {allgather_algo.instance.chunks})')

    if reducescatter_algo.topology.name != allgather_algo.topology.name:
        # TODO: improve this check to check actual structure, not just name
        raise ValueError(f'ReduceScatter and Allgather must have the same topology (got {reducescatter_algo.topology.name} and {allgather_algo.topology.name})')
    topo = reducescatter_algo.topology

    coll = allreduce(topo.num_nodes())

    steps = reducescatter_algo.steps + allgather_algo.steps
    instance = Instance(len(steps),
        extra_rounds=reducescatter_algo.instance.extra_rounds+allgather_algo.instance.extra_rounds,
        chunks=reducescatter_algo.instance.chunks)
    return Algorithm.make_implementation(coll, topo, instance, steps)