# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.topologies import dgx1
from msccl.collectives import gather, scatter
from msccl.strategies import solve_least_steps
from msccl.distributors.gather_scatter_alltoall import synthesize_gather_scatter_distributed_alltoall
from msccl.autosynth.registry import register_synthesis_plan
from msccl.ncclize import ncclize


def register_ndv2_plans():
    @register_synthesis_plan('alltoall', 'ndv2', sizes=('1MB', None), machines=lambda x: x >= 2)
    def synthesize_ndv2_relay_alltoall(machines):
        gather_coll = gather(8, 0)
        scatter_coll = scatter(8, 1)
        gather_algo = solve_least_steps(dgx1(), gather_coll)
        scatter_algo = solve_least_steps(dgx1(), scatter_coll)
        algo = synthesize_gather_scatter_distributed_alltoall(
            machines, gather_algo, scatter_algo)
        return ncclize(algo, instances=8)