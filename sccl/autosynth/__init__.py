# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.topologies.nvidia import nvlink_only
from sccl.autosynth.dgx1_relay_node_plan import DGX1RelayNodePlan
from sccl.ncclize import ncclize
import re, subprocess, tempfile, os, json, atexit, time

def init(verbose=False):
    env = _autosynth_assume_deterministic_z3_and_ompi(verbose)
    os.environ.update(env)
    return

    # The code below does not work in all usecases with PyTorch, due to mpi4py calling MPI_Init, which
    # some part of PyTorch cannot tolerate. The other way around would work, importing mpi4py after
    # torch.distributed has initialized, but currently the SCCL interpreter in NCCL cannot load new algorithms
    # after initialization. Once this dynamic loading support lands the code path below can be re-enabled.

    # Detect how this process was launched
    if 'LOCAL_RANK' in os.environ:
        # Either torch.distributed.run or legacy run with --use_env
        has_subprocesses = True
        world_size = int(os.environ['WORLD_SIZE'])
        is_mpi_process = int(os.environ['LOCAL_RANK']) == 0
        if verbose:
            print(f'SCCL: Found LOCAL_RANK in environment, torch.distributed.run (or launch with --use_env) detected.')
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", type=int)
        args, _ = parser.parse_known_args()
        if args.local_rank != None:
            # Legacy torch.distributed.launch without --use_env
            has_subprocesses = True
            world_size = int(os.environ['WORLD_SIZE'])
            is_mpi_process = args.local_rank == 0
            if verbose:
                print('SCCL: Found --local_rank N argument, legacy torch.distributed.launch without --use_env detected.')
        else:
            # Pure MPI
            has_subprocesses = False
            world_size = None
            is_mpi_process = True
            if verbose:
                print(f'SCCL: No launcher detected, assuming one MPI rank per process.')
    # Name environment file by parent PID, which will be shared between subprocesses for torch.distributed.(launch|run)
    env_file = f'/var/lock/sccl_autosynth_env.{os.getppid()}.lock'
    if is_mpi_process:
        # Synthesize on MPI rank 0 and distribute to all MPI processes
        env = _autosynth_and_get_env(world_size, verbose)
        # If there are non-MPI subprocesses, they get the environment through a temporary file
        if has_subprocesses:
            # Make sure the lock file doesn't exist yet
            if os.path.exists(env_file):
                raise RuntimeError(f'SCCL: Lock file already exists: {env_file}')
            # Broadcast algorithm to other subprocesses
            fd, private_file = tempfile.mkstemp()
            with open(fd, "w") as f:
                json.dump(env, f)
            os.rename(private_file, env_file)
            # Delete the environment file when the local MPI process exits
            atexit.register(os.remove, env_file)
    else:
        assert has_subprocesses
        # Wait until the environment file is available
        elapsed = 0
        while not os.path.exists(env_file):
            time.sleep(1)
            elapsed += 1
            if elapsed == 60:
                print(f'SCCL: Still waiting to read lock file {env_file}...')
        # Load the environment to set from the file
        with open(env_file, "r") as f:
            env = json.load(f)

    os.environ.update(env)

def ndv2_perm(verbose=True):
    machine = detect_machine(verbose)
    if machine[1] == None:
        return
    plan = select_synthesis_plan(machine)
    plan.local_rank_permutation()
    

def _autosynth_assume_deterministic_z3_and_ompi(verbose):
    rank = None
    if 'WORLD_SIZE' in os.environ:
        # We're in a PyTorch launcher compatible script
        world_size = int(os.environ['WORLD_SIZE'])
        if 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
    else:
        if not 'OMPI_COMM_WORLD_SIZE' in os.environ:
            print('SCCL: Could not detect world size. Please set either WORLD_SIZE or OMPI_COMM_WORLD_SIZE to total number of processes.')
            raise RuntimeError('Could not detect world size.')
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        if 'OMPI_COMM_WORLD_RANK' in os.environ:
            rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

    collective_names = ['Alltoall']
    if rank == 0:
        print(f'SCCL: Synthesizing algorithm(s) for {", ".join(collective_names)}...')
    
    machine = detect_machine(verbose)
    plan = select_synthesis_plan(machine)
    efs = []
    for name in collective_names:
        algo = plan.synthesize(world_size, name, verbose)
        efs.append(ncclize(algo, old_format=True, use_scratch=True, instances=8))

    tempdir = tempfile.mkdtemp()
    ef_files = []
    for name, ef in zip(collective_names, efs):
        ef_file = os.path.join(tempdir, f'{name}.xml')
        ef_files.append(ef_file)
        with open(ef_file, 'w') as f:
            f.write(ef)
        if verbose:
            print(f'SCCL: Wrote to {ef_file}')

    if len(ef_files) != 1:
        raise RuntimeError(f'Only a single algorithm is supported currently by the NCCL backend, but got {len(efs)}.')

    return {
        'SCCL_XML_FILE': ef_files[0],
        'NCCL_NET_SHARED_BUFFERS': '0',
        'NCCL_MIN_NCHANNELS': str(algo.nchannels)
    }

def _autosynth_and_get_env(world_size, verbose):
    try:
        from mpi4py import MPI
    except ImportError as e:
        print('SCCL: Please install the mpi4py package to use SCCL\'s automated init function.')
        raise e
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    if world_size == None:
        world_size = mpi_size

    collective_names = ['Alltoall']
    if mpi_rank == 0:
        print(f'SCCL: Synthesizing algorithm(s) for {", ".join(collective_names)}...')

    machine = detect_machine(verbose)
    plan = select_synthesis_plan(machine)
    names = comm.gather(machine[0], root=0)
    if mpi_rank == 0:
        for i in range(len(names) - 1):
            if names[i] != names[i+1]:
                raise RuntimeError(f'Rank {i} detected machine as {names[i]} but rank {i+1} detected machine as {names[i+1]}.')
        efs = []
        for name in collective_names:
            algo = plan.synthesize(world_size, name, verbose)
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
        if verbose:
            print(f'SCCL: Wrote to {ef_file}')

    if len(ef_files) != 1:
        raise RuntimeError(f'Only a single algorithm is supported currently by the NCCL backend, but got {len(efs)}.')

    perm = plan.local_rank_permutation()

    return {
        'SCCL_XML_FILE': ef_files[0],
        'CUDA_VISIBLE_DEVICES': ','.join(str(rank) for rank in perm)
    }

def detect_machine(verbose):
    machine = _detect_nvidia_machine(verbose)
    if machine != None:
        return machine
    return ('unknown', None)

def _detect_nvidia_machine(verbose):
    if verbose:
        print('SCCL: Checking for NVIDIA machines')
    try:
        smi_topo = subprocess.check_output(['nvidia-smi', 'topo', '-m']).decode("utf-8")
    except FileNotFoundError:
        if verbose:
            print('SCCL: nvidia-smi not found.')
        return None
    except subprocess.CalledProcessError:
        if verbose:
            print('SCCL: Found nvidia-smi, but got error.')
        return ('unknown', None)

    nvlink_topo = nvlink_only(smi_topo)

    if nvlink_topo.num_nodes() == 8: # DGX-1 and DGX A100 like nodes
        if verbose:
            print('SCCL: 8 GPUs, so looks like a DGX-1 or DGX A100.')
        if _is_one_host_ib_dgx1(smi_topo):
            return ('one_host_ib_dgx1', nvlink_topo)
        else:
            if verbose:
                print('SCCL: Unknown network configuration.')
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