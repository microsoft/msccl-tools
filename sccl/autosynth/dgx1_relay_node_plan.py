# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.topologies import dgx1
from sccl.collectives import gather, scatter
from sccl.strategies import solve_least_steps
from sccl.distributors.gather_scatter_alltoall import synthesize_gather_scatter_distributed_alltoall
from sccl.isomorphisms import find_isomorphisms
import re, subprocess, fcntl, tempfile, os, json

class DGX1RelayNodePlan:
    def __init__(self, local_topo):
        self.local_topo = local_topo

    def synthesize(self, world_size, collective_name, logging=False):
        if world_size % self.local_topo.num_nodes() != 0:
            raise RuntimeError('Local machine size does not evenly divide world size.')
        num_machines = world_size // self.local_topo.num_nodes()
        if collective_name == 'Alltoall':
            return self._synthesize_alltoall(num_machines, logging)

    def _synthesize_alltoall(self, num_machines, logging):
        outbound, inbound = self._select_root_nodes()
        gather_coll = gather(8, outbound)
        scatter_coll = scatter(8, inbound)
        gather_algo = solve_least_steps(dgx1(), gather_coll, logging=logging)
        scatter_algo = solve_least_steps(dgx1(), scatter_coll, logging=logging)
        return synthesize_gather_scatter_distributed_alltoall(num_machines, gather_algo, scatter_algo, logging)

    def _select_root_nodes(self):
        # TODO: is this always the right thing?
        return (0,1)

    def local_rank_permutation(self):
        isomorphisms = find_isomorphisms(dgx1(), self.local_topo)
        if len(isomorphisms) != 4:
            raise RuntimeError(f'Expected to find 4 isomorphisms to DGX1 topology, but found {len(isomorphisms)}.')
        return self._select_isomorphism(isomorphisms)

    def _select_isomorphism(self, isomorphisms):
        with open(os.path.join(tempfile.gettempdir(), 'sccl_autosynth_inspector_topo.lock'), "r+") as f:
            fcntl.lockf(f, fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(0)
                if size > 0:
                    return json.load(f)
                else:
                    print("Running inspector-topo to find the IB placement. This will take a minute...")
                    topo_detect = subprocess.run(['/usr/local/bin/inspector-topo'], capture_output=True, env={"CUDA_VISIBLE_DEIVCES":"0,1,2,3,4,5,6,7"})
                    if topo_detect.returncode != 0:
                        raise RuntimeError(f'inspector-topo had a failure:\n{topo_detect.stdout}\n{topo_detect.stderr}')
                    topo_detect_output = topo_detect.stdout.decode('utf-8')
                    g = re.search("GPU pair shared with NIC appears to be (\d) and (\d)", topo_detect_output)
                    if g is None:
                        raise RuntimeError(f'expected to detect a pair of GPUs connected to IB but something went wrong!')
                    ib_gpus = {int(g.group(1)), int(g.group(2))}
                    for iso in isomorphisms:
                        if len(ib_gpus.intersection({iso.nodes[0],iso.nodes[2]})) == 0:
                            nodes = iso.nodes
                            json.dump(nodes, f)
                            return nodes
                    raise RuntimeError(f'expected an isomorphism to match our expectation but none of them did!')
            finally:
                fcntl.lockf(f, fcntl.LOCK_UN)
