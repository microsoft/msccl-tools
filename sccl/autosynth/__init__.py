# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.topologies import dgx1
from sccl.isomorphisms import find_isomorphisms
from sccl.autosynth.registry import synthesis_plans
import re
import subprocess
import fcntl
import os
import humanfriendly

from sccl.autosynth.dgx1_plans import register_dgx1_plans
from sccl.autosynth.a100_plans import register_a100_plans
register_dgx1_plans()
register_a100_plans()


def init(num_machines, machine_type, *collectives):
    plans_and_sizes = []
    for collective in collectives:
        name, size = collective
        if isinstance(size, str):
            size = humanfriendly.parse_size(size)
        candidates = synthesis_plans[(name, machine_type)]
        valid_candidates = filter(
            _candidate_filter(num_machines, size), candidates)
        sorted_candidates = sorted(valid_candidates, key=_candidate_sort_key)
        description = f'{name} with size {humanfriendly.format_size(size)}'
        if len(sorted_candidates) == 0:
            print(
                f'SCCL: No plan found for {description}. Falling back to NCCL baseline.')
        else:
            desc, plan, _, _, _ = sorted_candidates[-1]
            print(f'SCCL: Plan for {description} is {desc}')
            plans_and_sizes.append((plan, size))

    envs = {}
    for plan, size in plans_and_sizes:
        path, env = plan(num_machines, size)
        if 'SCCL_XML_FILE' in envs:
            envs['SCCL_XML_FILE'] += ',' + path
        else:
            envs['SCCL_XML_FILE'] = path
        envs.update(env)

    os.environ.update(envs)


def _candidate_filter(m, s):
    def fun(candidate):
        _, _, machines, size_ranges, _ = candidate
        size_matches = any(map(lambda x: x[0] <= s and s <= x[1], size_ranges))
        return size_matches and machines(m)
    return fun


def _candidate_sort_key(candidate):
    _, _, _, _, priority = candidate
    return priority


def ndv2_perm(self): # pragma: no cover
    # This function is used in a hacky way right now. The sccl_ndv2_launcher.sh
    # relies on the side effect of _select_isomorphism creating the lock file,
    # which is read by the script after calling this function, so the return
    # value does't currently get used. If you make changes, please fix or update
    # sccl_ndv2_launcher.sh accordingly.
    isomorphisms = find_isomorphisms(dgx1(), self.local_topo)
    if len(isomorphisms) != 4:
        raise RuntimeError(
            f'Expected to find 4 isomorphisms to DGX1 topology, but found {len(isomorphisms)}.')
    return _select_isomorphism(isomorphisms)


def _select_isomorphism(self, isomorphisms, verbose=True): # pragma: no cover
    with open('/var/lock/sccl_autosynth_inspector_topo.lock', "a+") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        try:
            f.seek(0, 2)
            size = f.tell()
            if size > 0:
                f.seek(0)
                order = f.read()
                if verbose:
                    print(f'SCCL: Read IB placement from {f.name}')
                return order
            else:
                print(
                    'SCCL: Running inspector-topo to find the IB placement. This will take a couple of minutes...')
                topo_detect = subprocess.run(
                    ['/usr/local/bin/inspector-topo'], capture_output=True, env={"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"})
                print('SCCL: Finished running inspector-topo. Finding the permutaion.')
                if topo_detect.returncode != 0:
                    raise RuntimeError(
                        f'inspector-topo had a failure:\n{topo_detect.stdout}\n{topo_detect.stderr}')
                topo_detect_output = topo_detect.stdout.decode('utf-8')
                g = re.search(
                    "GPU pair shared with NIC appears to be (\d) and (\d)", topo_detect_output)
                if g is None:
                    raise RuntimeError(
                        f'expected to detect a pair of GPUs connected to IB but something went wrong!')
                ib_gpus = {int(g.group(1)), int(g.group(2))}
                for iso in isomorphisms:
                    if len(ib_gpus.intersection({iso.nodes[0], iso.nodes[2]})) == 0:
                        nodes = iso.nodes
                        order = ",".join(str(rank) for rank in nodes)
                        f.write(order)
                        f.flush()
                        if verbose:
                            print(f'SCCL: Wrote IB placement to {f.name}')
                        return order
                raise RuntimeError(
                    f'expected an isomorphism to match our expectation but none of them did!')
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)
