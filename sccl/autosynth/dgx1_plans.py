# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.topologies import dgx1
from sccl.collectives import gather, scatter
from sccl.strategies import solve_least_steps
from sccl.distributors.gather_scatter_alltoall import synthesize_gather_scatter_distributed_alltoall
from sccl.autosynth.registry import register_synthesis_plan
from sccl.ncclize import ncclize


def register_dgx1_plans():
    @register_synthesis_plan('alltoall', 'dgx1', machines=lambda x: x >= 2)
    def synthesize_dgx1_relay_alltoall(machines, size):
        gather_coll = gather(8, 0)
        scatter_coll = scatter(8, 1)
        gather_algo = solve_least_steps(dgx1(), gather_coll)
        scatter_algo = solve_least_steps(dgx1(), scatter_coll)
        algo = synthesize_gather_scatter_distributed_alltoall(
            machines, gather_algo, scatter_algo)
        return ncclize(algo, old_format=True, use_scratch=True, instances=8)