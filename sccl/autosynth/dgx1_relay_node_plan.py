# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.topologies import dgx1
from sccl.collectives import gather, scatter
from sccl.strategies import solve_least_steps
from sccl.distributors.gather_scatter_alltoall import synthesize_gather_scatter_distributed_alltoall
from sccl.isomorphisms import find_isomorphisms

class DGX1RelayNodePlan:
    def __init__(self, local_topo):
        self.local_topo = local_topo
    
    def synthesize(self, world_size, collective_names, logging=False):
        if world_size % self.local_topo.num_nodes() != 0:
            raise RuntimeError('Local machine size does not evenly divide world size.')
        num_machines = world_size // self.local_topo.num_nodes()
        for name in collective_names:
            if name == 'Alltoall':
                yield self._synthesize_alltoall(num_machines, logging)
    
    def _synthesize_alltoall(self, num_machines, logging):
        outbound, inbound = self._select_root_nodes()
        gather_coll = gather(8, outbound)
        scatter_coll = scatter(8, inbound)
        gather_algo = solve_least_steps(dgx1(), gather_coll, logging=logging)
        scatter_algo = solve_least_steps(dgx1(), scatter_coll, logging=logging)
        synthesize_gather_scatter_distributed_alltoall(num_machines, gather_algo, scatter_algo, logging)

    def _select_root_nodes(self):
        # TODO: is this always the right thing?
        return (0,1)

    def local_rank_permutation(self):
        isomorphisms = find_isomorphisms(dgx1(), self.local_topo)
        if len(isomorphisms) != 4:
            raise RuntimeError(f'Expected to find 4 isomorphisms to DGX1 topology, but found {len(isomorphisms)}.')
        return self._select_isomorphism(isomorphisms)

    def _select_isomorphism(self, isomorphisms):
        # TODO: do the microbenchmarking
        return isomorphisms[0]
