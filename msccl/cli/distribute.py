# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.distributors import *
from .known_distributed_topologies import KnownDistributedTopologies
from .known_topologies import KnownTopologies
from .common import *

def make_distributors(cmd_parsers):
    handler_funcs = []
    handler_funcs.append(make_handle_greedy_alltoall)
    handler_funcs.append(make_handle_gather_scatter_alltoall)
    handler_funcs.append(make_handle_create_subproblem_distributed_alltoall)
    handler_funcs.append(make_handle_distribute_alltoall_stitch_subproblem)

    return make_cmd_category(cmd_parsers, 'distribute', 'distributor', handler_funcs)

def make_handle_greedy_alltoall(cmd_parsers):
    name = 'alltoall-greedy'
    cmd = cmd_parsers.add_parser(name)
    read_algorithm = add_input_algorithm(cmd)
    distributed_topologies = KnownDistributedTopologies(cmd)
    validate_output_args, output_handler = add_output_algorithm(cmd)

    def handle(args, command):
        if command != name:
            return False

        input_algorithm = read_algorithm(args)
        validate_output_args(args)
        topology = distributed_topologies.create(args, input_algorithm.topology)
        algo = synthesize_greedy_distributed_alltoall(topology, input_algorithm, logging=True)
        output_handler(args, algo)
        return True

    return handle

def make_handle_gather_scatter_alltoall(cmd_parsers):
    name = 'alltoall-gather-scatter'
    cmd = cmd_parsers.add_parser(name)
    read_gather_algorithm = add_input_algorithm(cmd, name='gather')
    read_scatter_algorithm = add_input_algorithm(cmd, name='scatter')
    cmd.add_argument('--copies', type=int, metavar='N', required=True, help='copies of the local topology to be made')
    cmd.add_argument('-bw', '--remote-bandwidth', type=int, default=1, help='remote bandwidth', metavar='N')
    validate_output_args, output_handler = add_output_algorithm(cmd)

    def handle(args, command):
        if command != name:
            return False

        gather_algorithm = read_gather_algorithm(args)
        scatter_algorithm = read_scatter_algorithm(args)
        validate_output_args(args)
        algo = synthesize_gather_scatter_distributed_alltoall(args.copies, gather_algorithm, scatter_algorithm, args.remote_bandwidth, logging=True)
        output_handler(args, algo)
        return True

    return handle

def make_handle_create_subproblem_distributed_alltoall(cmd_parsers):
    name = 'alltoall-create-subproblem'
    cmd = cmd_parsers.add_parser(name)
    topologies = KnownTopologies(cmd)
    cmd.add_argument('--copies', type=int, metavar='N', required=True, help='copies of the local topology to be made')
    cmd.add_argument('--relay-nodes', type=int, nargs='+', default=[0], help='relay nodes')
    cmd.add_argument('-bw', '--remote-bandwidth', type=int, default=1, help='remote bandwidth', metavar='N')
    cmd.add_argument('--share-bandwidth', action='store_true', help='share local bandwidth between relay nodes')
    validate_output_args, output_handler = add_output_msccl_objects(cmd)

    def handle(args, command):
        if command != name:
            return False

        local_topology = topologies.create(args)
        validate_output_args(args)

        collective, topology = make_alltoall_subproblem_collective_and_topology(local_topology, args.copies, args.relay_nodes, args.remote_bandwidth, args.share_bandwidth)

        output_handler(args, collective, collective.name)
        output_handler(args, topology, topology.name)
        return True

    return handle
 
def make_handle_distribute_alltoall_stitch_subproblem(cmd_parsers):
    name = 'alltoall-stitch-subproblem'
    cmd = cmd_parsers.add_parser(name)
    read_subproblem_algorithm = add_input_algorithm(cmd)
    cmd.add_argument('--copies', type=int, metavar='N', required=True, help='copies of the local topology made for the subproblem')
    validate_output_args, output_handler = add_output_algorithm(cmd)

    def handle(args, command):
        if command != name:
            return False

        subproblem_algorithm = read_subproblem_algorithm(args)
        validate_output_args(args)
        algo = synthesize_alltoall_subproblem(subproblem_algorithm, args.copies, logging=True)
        output_handler(args, algo)
        return True

    return handle