# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.topologies.nvidia import nvlink_only
from sccl.autosynth.dgx1_relay_node_plan import DGX1RelayNodePlan
import subprocess
import re

def init(logging=False):
    try:
        from mpi4py import MPI
    except ImportError as e:
        print('Please install the mpi4py package to use SCCL autosynth.')
        raise e
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    machine = detect_machine()
    names = comm.gather(machine[0], root=0)
    if rank == 0:
        for i in range(len(names) - 1):
            if names[i] != names[i+1]:
                raise RuntimeError(f'Rank {i} detected machine as {names[i]} but rank {i+1} detected machine as {names[i+1]}.')
        plan = select_synthesis_plan(machine)
        for algo in plan.synthesize(size, ['Alltoall'], logging):
            pass

def detect_machine():
    machine = _detect_nvidia_machine()
    if machine != None:
        return machine
    return ('unknown', None)

def _detect_nvidia_machine():
    try:
        smi_topo = subprocess.check_output(['nvidia-smi', 'topo', '-m']).decode("utf-8")
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError:
        return ('unknown', None)

    nvlink_topo = nvlink_only(smi_topo)

    if nvlink_topo.num_nodes == 8: # DGX-1 and DGX A100 like nodes
        if _is_one_host_ib_dgx1():
            return ('one_host_ib_dgx1', nvlink_topo)

def _is_one_host_ib_dgx1(smi_topo):
    ib_host = re.findall('^mlx\\d_\\d(\s+NODE)*\s+X(\s+NODE)*&', smi_topo, re.MULTILINE)
    ib_any = re.findall('^mlx\\d_\\d.*&', smi_topo, re.MULTILINE)
    return len(ib_host) == 1 and len(ib_any) == 1

def select_synthesis_plan(machine):
    machine_name, machine_info = machine
    if machine_name == 'one_host_ib_dgx1':
        return DGX1RelayNodePlan(machine_info)
