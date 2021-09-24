# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.topologies import dgx1
from sccl.isomorphisms import find_isomorphisms
from sccl.autosynth.registry import synthesis_plans
from lxml import etree as ET
import re
import subprocess
import fcntl
import os
import math
import tempfile
import humanfriendly

from sccl.autosynth.dgx1_plans import register_dgx1_plans
from sccl.autosynth.a100_plans import register_a100_plans
register_dgx1_plans()
register_a100_plans()


def init(num_machines, machine_type, *collectives):
    # Collect and sort all plans that match the collectives and sizes given by the user.
    selected_plans = {}
    for collective in collectives:
        name, sizes = collective
        if isinstance(sizes, tuple):
            lower, upper = sizes
            if isinstance(lower, str):
                lower = humanfriendly.parse_size(lower)
            if isinstance(upper, str):
                upper = humanfriendly.parse_size(upper)
            if upper == None:
                upper = math.inf
            sizes = (lower, upper)
        else:
            if isinstance(sizes, str):
                sizes = humanfriendly.parse_size(sizes)
            sizes = (sizes, sizes)
        candidates = synthesis_plans[(name, machine_type)]
        selected_plans[name] = _select_plans(name, candidates, num_machines, sizes)

    if len(selected_plans) > 0:
        # Execute the plans to find or synthesize the algorithms and format them in the XML format expected by SCCL-RT.
        algos_elem = ET.Element('sccl_algos')
        max_min_channels = 0
        for collective_name, plans in selected_plans.items():
            for plan, params in plans:
                path = plan(num_machines)
                min_channels = _extract_min_channels(path)
                # Skip the algorithm if minimum channels could not be determined (corrupted XML for example)
                if min_channels:
                    max_min_channels = max(max_min_channels, min_channels)

                    load_elem = ET.SubElement(algos_elem, 'load')
                    load_elem.set('path', path)
                    minsize, maxsize, proto = params
                    if minsize != 0:
                        load_elem.set('minsize', str(minsize))
                    if maxsize != math.inf:
                        load_elem.set('maxsize', str(maxsize+1))
                    load_elem.set('proto', proto)
        ET.indent(algos_elem, space='  ')
        
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as f:
            f.write(ET.tostring(algos_elem, encoding='unicode'))
        os.environ.update({
            'SCCL_CONFIG': path,
            'NCCL_MIN_NCHANNELS': str(max_min_channels),
            'NCCL_NET_SHARED_BUFFERS': '0'
        })
    else:
        print(f'SCCL: No algorithms were selected.')


def _select_plans(name, candidates, num_machines, sizes):
    candidate_intervals = [((0, math.inf), [])]
    valid_candidates = list(filter(lambda x: x[2](num_machines), candidates))
    for candidate in valid_candidates:
        csizes = candidate[3]
        # Skip candidate if it does not overlap with user provided sizes
        if csizes[0] > sizes[1] or sizes[0] > csizes[1]:
            continue
        i = 0
        while i < len(candidate_intervals):
            ival = candidate_intervals[i]
            isizes = ival[0]
            if isizes[1] < csizes[0]:
                i += 1
                continue
            if isizes[0] > csizes[1]:
                break
            if isizes[0] < csizes[0]:
                del candidate_intervals[i]
                candidate_intervals.insert(i, ((csizes[0], isizes[1]), ival[1]))
                candidate_intervals.insert(i, ((isizes[0], csizes[0]-1), ival[1].copy()))
                i += 1
                continue
            if isizes[1] > csizes [1]:
                del candidate_intervals[i]
                candidate_intervals.insert(i, ((csizes[1]+1, isizes[1]), ival[1]))
                candidate_intervals.insert(i, ((isizes[0], csizes[1]), ival[1] + [candidate]))
                break
            ival[1].append(candidate)
            csizes = (isizes[1]+1,csizes[1])
            if csizes[0] > csizes[1]:
                break
    results = []
    for isizes, candidates in candidate_intervals:
        # Skip interval if it does not overlap with user provided sizes
        if isizes[0] > sizes[1] or sizes[0] > isizes[1]:
            continue
        sorted_candidates = sorted(candidates, key=_candidate_sort_key)
        description = f'{name} with sizes from {humanfriendly.format_size(isizes[0])} to {humanfriendly.format_size(isizes[1])}'
        if len(sorted_candidates) == 0:
            print(f'SCCL: No plan found for {description}. Falling back to NCCL baseline.')
        else:
            desc, plan, _, _, proto, _ = sorted_candidates[-1]
            print(f'SCCL: Plan for {description} is {desc} with {proto} protocol.')
            if len(results) > 0 and plan == results[-1][0] and isizes[0] == results[-1][1][1] + 1 and proto == results[-1][1][2]:
                results[-1][1][1] = isizes[1]
            else:
                results.append((plan, [isizes[0], isizes[1], proto]))
    return results


def _candidate_sort_key(candidate):
    _, _, _, _, _, priority = candidate
    return priority


def _extract_min_channels(path):
    algo_pattern = re.compile('<algo[^>]*>')
    nchannels_pattern = re.compile('nchannels=["\'](\\d+)["\']')
    with open(path) as f:
        # Try with the first line
        first_line = f.readline()
        match = algo_pattern.search(first_line)
        if match:
            tag_match = nchannels_pattern.search(match.group(0))
            if not tag_match:
                print(f'SCCL: Skipping algorithm, could not read nchannels from <algo/> tag in {path}')
                return None
            return int(tag_match.group(1))
        # Try again with the whole file
        f.seek(0)
        whole_file = f.read()
        match = algo_pattern.search(whole_file)
        if match:
            tag_match = nchannels_pattern.search(match.group(0))
            if not tag_match:
                print(f'SCCL: Skipping algorithm, could not read nchannels from <algo/> tag in {path}')
                return None
            return int(tag_match.group(1))
        else:
            print(f'SCCL: Skipping algorithm, could not find <algo/> tag in {path}')
            return None


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
                    'GPU pair shared with NIC appears to be (\\d) and (\\d)', topo_detect_output)
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
