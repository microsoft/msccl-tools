# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.topologies import dgx1, dgx_a100, nvlink_only
from msccl.isomorphisms import find_isomorphisms
from msccl.autosynth.registry import synthesis_plans
from lxml import etree as ET
import re
import subprocess
import fcntl
import os
import math
import tempfile
import humanfriendly
from tabulate import tabulate
from enum import Enum

from msccl.autosynth.ndv2_plans import register_ndv2_plans
from msccl.autosynth.ndv4_plans import register_ndv4_plans
register_ndv2_plans()
register_ndv4_plans()


class Collective(Enum):
    allreduce = 'allreduce'
    allgather = 'allgather'
    reduce = 'reduce'
    broadcast = 'broadcast'
    alltoall = 'alltoall'
    reduce_scatter = 'reduce_scatter'

    def __str__(self):
        return self.value


def init(machine_type, num_machines, *collectives):
    # first detect the machine type in case auto was passed in
    if machine_type == "auto":
        nvlink_matrix = nvlink_only()
        isomorphisms = find_isomorphisms(dgx1(), nvlink_matrix)
        if len(isomorphisms) == 4:
            machine_type = "ndv2"
        elif nvlink_matrix.links == dgx_a100().links:
            machine_type = "ndv4"
        else:
            print(f'Did not recognize the SKU type automatically. If you are sure about the SKU, try replacing "auto" with your explicit SKU name. Falling back to NCCL.')
            return
        print(f"The auto-detected SKU is a {machine_type}.")

    # Collect and sort all plans that match the collectives and sizes given by the user.
    selected_plans = {}
    for collective in collectives:
        name, sizes = collective
        if isinstance(name, Collective):
            name = str(name)
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
            sizes = (sizes, sizes+1)
        candidates = synthesis_plans[(name, machine_type)]
        plans = _select_plans(name, candidates, num_machines, sizes)
        if len(plans) > 0:
            selected_plans[name] = plans

    # Execute the plans to find or synthesize the algorithms and format them in the XML format expected by MSCCL-RT.
    algos_elem = ET.Element('msccl_algos')
    any_selected = False
    for collective_name, plans in selected_plans.items():
        for plan, params in plans:
            path = plan(num_machines)
            load_elem = ET.SubElement(algos_elem, 'load')
            load_elem.set('path', path)
            minsize, maxsize, proto = params
            if minsize != 0:
                load_elem.set('minBytes', str(minsize))
            if maxsize != math.inf:
                load_elem.set('maxBytes', str(maxsize))
            load_elem.set('proto', proto)
            any_selected = True
    ET.indent(algos_elem, space='  ')
        
    if any_selected:
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as f:
            f.write(ET.tostring(algos_elem, encoding='unicode'))

        # Set environment variables
        env = {
            'MSCCL_CONFIG': path,
        }
        if 'NCCL_ALGO' in os.environ and os.environ['NCCL_ALGO'] != '':
            existing_algos = os.environ['NCCL_ALGO']
            if 'MSCCL' not in existing_algos.split(','):
                os.environ['NCCL_ALGO'] = 'MSCCL,' + existing_algos
        else:
            env['NCCL_ALGO'] = 'MSCCL,RING,TREE'
        if machine_type == 'ndv4' and num_machines >= 8 and 'alltoall' in selected_plans:
            print(f'MSCCL: Setting NCCL_IB_AR_THRESHOLD=0 (reason: alltoall and at least 16 ndv4 machines)')
            env['NCCL_IB_AR_THRESHOLD'] = '0'
        if machine_type == 'ndv4':
            print(f'MSCCL: Setting relaxed orderin, topo file and visible devices order')
            env['NCCL_IB_PCI_RELAXED_ORDERING'] = '1'
            env['NCCL_TOPO_FILE'] = '/opt/msft/topo.xml'
            env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
        os.environ.update(env)
    else:
        print(f'MSCCL: No algorithms were selected.')


def _format_size(size):
    if size != math.inf:
        return humanfriendly.format_size(size)
    else:
        return 'infinity'


def _select_plans(name, candidates, num_machines, sizes):
    candidate_intervals = [((0, math.inf), [])]
    valid_candidates = list(filter(lambda x: x[2](num_machines), candidates))
    for candidate in valid_candidates:
        csizes = candidate[3]
        # Skip candidate if it does not overlap with user provided sizes
        if csizes[0] >= sizes[1] or sizes[0] >= csizes[1]:
            continue
        i = 0
        while i < len(candidate_intervals):
            ival = candidate_intervals[i]
            isizes = ival[0]
            if isizes[1] <= csizes[0]:
                i += 1
                continue
            if isizes[0] >= csizes[1]:
                break
            if isizes[0] < csizes[0]:
                del candidate_intervals[i]
                candidate_intervals.insert(i, ((csizes[0], isizes[1]), ival[1]))
                candidate_intervals.insert(i, ((isizes[0], csizes[0]), ival[1].copy()))
                i += 1
                continue
            if isizes[1] > csizes [1]:
                del candidate_intervals[i]
                candidate_intervals.insert(i, ((csizes[1], isizes[1]), ival[1]))
                candidate_intervals.insert(i, ((isizes[0], csizes[1]), ival[1] + [candidate]))
                break
            ival[1].append(candidate)
            csizes = (isizes[1],csizes[1])
            if csizes[0] >= csizes[1]:
                break
            if csizes[0] == math.inf:
                break
    results = []
    for isizes, candidates in candidate_intervals:
        # Skip interval if it does not overlap with user provided sizes
        if isizes[0] >= sizes[1] or sizes[0] >= isizes[1]:
            continue
        sorted_candidates = sorted(candidates, key=_candidate_sort_key)
        description = f'{name} with sizes from {_format_size(isizes[0])} to {_format_size(isizes[1])}'
        if len(sorted_candidates) == 0:
            print(f'MSCCL: No plan found for {description}. Falling back to NCCL baseline.')
        else:
            desc, plan, _, _, proto, _ = sorted_candidates[-1]
            print(f'MSCCL: Plan for {description} is {desc} with {proto} protocol.')
            if len(results) > 0 and plan == results[-1][0] and isizes[0] == results[-1][1][1] + 1 and proto == results[-1][1][2]:
                results[-1][1][1] = isizes[1]
            else:
                results.append((plan, [isizes[0], isizes[1], proto]))
    return results


def _candidate_sort_key(candidate):
    _, _, _, _, _, priority = candidate
    return priority


def ndv2_perm(): # pragma: no cover
    # This function is used in a hacky way right now. The msccl_ndv2_launcher.sh
    # relies on the side effect of _select_isomorphism creating the lock file,
    # which is read by the script after calling this function, so the return
    # value does't currently get used. If you make changes, please fix or update
    # msccl_ndv2_launcher.sh accordingly.
    isomorphisms = find_isomorphisms(dgx1(), nvlink_only())
    if len(isomorphisms) != 4:
        raise RuntimeError(
            f'Expected to find 4 isomorphisms to DGX1 topology, but found {len(isomorphisms)}.')
    return _select_isomorphism(isomorphisms)


def _select_isomorphism(isomorphisms, verbose=True): # pragma: no cover
    with open('/var/lock/msccl_autosynth_inspector_topo.lock', "a+") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        try:
            f.seek(0, 2)
            size = f.tell()
            if size > 0:
                f.seek(0)
                order = f.read()
                if verbose:
                    print(f'MSCCL: Read IB placement from {f.name}')
                return order
            else:
                print(
                    'MSCCL: Running inspector-topo to find the IB placement. This will take a couple of minutes...')
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
                topo_detect = subprocess.run(
                    ['/usr/local/bin/inspector-topo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
                print('MSCCL: Finished running inspector-topo. Finding the permutaion.')
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
                            print(f'MSCCL: Wrote IB placement to {f.name}')
                        return order
                raise RuntimeError(
                    f'expected an isomorphism to match our expectation but none of them did!')
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


_max_described_machines = 2048
def _describe_machines(machines):
    ranges = []
    lower = None
    for i in range(_max_described_machines):
        if machines(i):
            if lower is None:
                lower = i
        else:
            if lower is not None:
                if lower == i-1:
                    ranges.append(str(i-1))
                else:
                    ranges.append(f'{lower}-{i-1}')
                lower = None
    if lower is not None:
        ranges.append(f'>={lower}')
    if len(ranges) > 0:
        return ','.join(ranges)
    else:
        return '???'


def _list_plan_parameters():
    headers = ['Machine', 'Collective', '# machines', 'From', 'To', 'Protocol', 'Priority', 'Plan name']
    rows = []
    for key, plans in synthesis_plans.items():
        collective, machine_type = key
        for name, function, machines, (low, high), protocol, priority in plans:
            # First tuple is the key to sort by, second is the actual columns
            rows.append(((machine_type,collective,low,high,protocol,priority,name),
                (machine_type, collective, _describe_machines(machines), _format_size(low), _format_size(high), protocol, priority, name)))
    rows = [columns for _, columns in sorted(rows, key=lambda x: x[0])]
    return headers, rows


def tabulate_plans():
    headers, rows = _list_plan_parameters()
    return tabulate(rows, headers=headers, tablefmt='github')


def print_plans():
    print(tabulate_plans())
