# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.topologies.nvidia import nvlink_only
from sccl.autosynth.dgx1_relay_node_plan import DGX1RelayNodePlan
from sccl.ncclize import ncclize
import re, subprocess, tempfile, os, json, atexit, time

def init(logging=True):
    if 'LOCAL_WORLD_SIZE' in os.environ:
        has_subprocesses = True
        world_size = os.environ['WORLD_SIZE']
        is_mpi_process = os.environ['LOCAL_RANK'] == 0
        if logging:
            print(f'SCCL: Found LOCAL_WORLD_SIZE in environment, torch.distributed.run detected, with {os.environ["LOCAL_WORLD_SIZE"]} subprocesses per machine.')
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", type=int)
        args = parser.parse_known_args()
        if args.local_rank != None:
            has_subprocesses = True
            world_size = os.environ['WORLD_SIZE']
            is_mpi_process = args.local_rank == True
            if logging:
                print('SCCL: Found --local_rank N argument, legacy torch.distributed.launch detected, assuming one subprocess per GPU.')
        else:
            has_subprocesses = False
            world_size = None
            is_mpi_process = True
            if logging:
                print(f'SCCL: No launcher detected, assuming one MPI rank per process.')
    # Name environment file by parent PID, which will be shared between subprocesses for torch.distributed.(launch|run)
    env_file = os.path.join(tempfile.gettempdir(), f'sccl_autosynth_env.{os.getppid()}.lock')
    if is_mpi_process:
        env = _autosynth_and_get_env(world_size, logging)
        if has_subprocesses:
            # Make sure the lock file doesn't exist yet
            if os.path.exists(env_file):
                raise RuntimeError(f'SCCL: Lock file already exists: {env_file}')
            # Broadcast algorithm to other subprocesses
            with open(env_file, "w") as f:
                json.dump(env, f)
            # Delete the environment file at local rank 0 exit
            atexit.register(os.remove(), env_file)
    else:
        assert has_subprocesses
        # Wait until the environment file is available
        elapsed = 0
        while not os.path.exists(env_file):
            time.sleep(1)
            elapsed += 1
            if elapsed == 10 and logging:
                print(f'SCCL: Still waiting to read lock file {env_file}...')
        # Load the environment to set from the file
        with open(env_file, "r") as f:
            env = json.load(f)

    os.environ.update(env)

    if logging:
        print('SCCL: Algorithms installed.')

def _autosynth_and_get_env(world_size, logging):
    try:
        from mpi4py import MPI
    except ImportError as e:
        print('Please install the mpi4py package to use SCCL autosynth.')
        raise e
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    if world_size == None:
        world_size = mpi_size

    collective_names = ['Alltoall']

    machine = detect_machine(logging)
    plan = select_synthesis_plan(machine)
    names = comm.gather(machine[0], root=0)
    if mpi_rank == 0:
        for i in range(len(names) - 1):
            if names[i] != names[i+1]:
                raise RuntimeError(f'Rank {i} detected machine as {names[i]} but rank {i+1} detected machine as {names[i+1]}.')
        efs = []
        for name in collective_names:
            algo = plan.synthesize(world_size, name, logging)
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
