# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.topologies.nvidia import nvlink_only
from sccl.autosynth.dgx1_relay_node_plan import DGX1RelayNodePlan
from sccl.ncclize import ncclize
import re, subprocess, fcntl, tempfile, os, json, glob

def init(logging=False, torch_distributed_launch_hack=False):
    if torch_distributed_launch_hack:
        with open(os.path.join(tempfile.gettempdir(), 'sccl_autosynth_env.lock'), "w+") as f:
            fcntl.lockf(f, fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(0)
                if size > 0:
                    env = json.load(f)
                else:
                    env = _autosynth_and_get_env(logging)
                    json.dump(env, f)
            finally:
                fcntl.lockf(f, fcntl.LOCK_UN)
    else:
        env = _autosynth_and_get_env(logging)

    os.environ.update(env)

def _autosynth_and_get_env(logging):
    try:
        from mpi4py import MPI
    except ImportError as e:
        print('Please install the mpi4py package to use SCCL autosynth.')
        raise e
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    collective_names = ['Alltoall']

    machine = detect_machine(logging)
    plan = select_synthesis_plan(machine)
    names = comm.gather(machine[0], root=0)
    if rank == 0:
        for i in range(len(names) - 1):
            if names[i] != names[i+1]:
                raise RuntimeError(f'Rank {i} detected machine as {names[i]} but rank {i+1} detected machine as {names[i+1]}.')
        efs = []
        for name in collective_names:
            algo = plan.synthesize(size, name, logging)
            efs.append(ncclize(algo, old_format=True, use_scratch=True))
    else:
        efs = None
    efs = comm.bcast(efs, root=0)

    tempdir = tempfile.mkdtemp()
    ef_files = []
    for name, ef in zip(collective_names, efs):
        ef_file = os.path.join(tempdir, f'{name}.xml')
        ef_files.append(ef_file)
        with open(ef_file, 'w') as f:
            f.write(ef)
        if logging:
            print(f'Wrote to {ef_file}')

    if len(ef_files) != 1:
        raise RuntimeError(f'Only a single algorithm is supported currently by the NCCL backend, but got {len(efs)}.')

    perm = plan.local_rank_permutation()

    return {
        'SCCL_XML_FILE': ef_files[0],
        'CUDA_VISIBLE_DEVICES': ','.join(str(rank) for rank in perm)
    }

def detect_machine(logging):
    machine = _detect_nvidia_machine(logging)
    if machine != None:
        return machine
    return ('unknown', None)

def _detect_nvidia_machine(logging):
    if logging:
        print('Checking for NVIDIA machines')
    try:
        smi_topo = subprocess.check_output(['nvidia-smi', 'topo', '-m']).decode("utf-8")
    except FileNotFoundError:
        if logging:
            print('nvidia-smi not found.')
        return None
    except subprocess.CalledProcessError:
        if logging:
            print('Found nvidia-smi, but got error.')
        return ('unknown', None)

    nvlink_topo = nvlink_only(smi_topo)

    if nvlink_topo.num_nodes() == 8: # DGX-1 and DGX A100 like nodes
        if logging:
            print('8 GPUs, so looks like a DGX-1 or DGX A100.')
        if _is_one_host_ib_dgx1(smi_topo):
            return ('one_host_ib_dgx1', nvlink_topo)
        else:
            if logging:
                print('Unknown network configuration.')
    return ('unknown', None)

def _is_one_host_ib_dgx1(smi_topo):
    ib_host = re.findall('^mlx\\d_\\d(?:\s+NODE)*\s+X(?:\s+NODE)*\s+$', smi_topo, re.MULTILINE)
    ib_any = re.findall('^mlx\\d_\\d.*$', smi_topo, re.MULTILINE)
    return len(ib_host) == 1 and len(ib_any) == 1

def select_synthesis_plan(machine):
    machine_name, machine_info = machine
    if machine_name == 'one_host_ib_dgx1':
        return DGX1RelayNodePlan(machine_info)
    else:
        raise RuntimeError(f'Unhandled machine type {machine_name}.')
