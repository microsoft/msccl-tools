# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import msccl.strategies as strategies
from .known_topologies import KnownTopologies
from .known_collectives import KnownCollectives
from .common import *

def make_solvers(cmd_parsers):
    handler_funcs = []
    handler_funcs.append(make_handle_solve_instance)
    handler_funcs.append(make_handle_solve_least_steps)
    handler_funcs.append(make_handle_solve_pareto_optimal)

    return make_cmd_category(cmd_parsers, 'solve', 'solver', handler_funcs)

def _make_handle_strategy(cmd_parsers, name, invoke, take_steps = True):
    cmd = cmd_parsers.add_parser(name)
    instance_handler = add_instance(cmd, take_steps=take_steps)
    topologies = KnownTopologies(cmd)
    collectives = KnownCollectives(cmd)
    validate_output_args, output_handler = add_output_algorithm(cmd)

    def handle(args, command):
        if command != name:
            return False

        validate_output_args(args)
        topology = topologies.create(args)
        collective = collectives.create(args, topology.num_nodes())
        instance = instance_handler(args)
        algo = invoke(args, topology, collective, instance)
        output_handler(args, algo)
        return True
    
    return cmd, handle

def make_handle_solve_instance(cmd_parsers):
    def invoke(args, topology, collective, instance):
        return strategies.solve_instance(topology, collective, instance, logging=True)

    cmd, handle = _make_handle_strategy(cmd_parsers, 'instance', invoke)
    return handle

def make_handle_solve_least_steps(cmd_parsers):
    def invoke(args, topology, collective, instance):
        return strategies.solve_least_steps(topology, collective, args.initial_steps, instance, logging=True)

    cmd, handle = _make_handle_strategy(cmd_parsers, 'least-steps', invoke, take_steps=False)
    cmd.add_argument('--initial-steps', type=int, default=1, metavar='N')
    return handle

def make_handle_solve_pareto_optimal(cmd_parsers):
    name = 'pareto-optimal'
    cmd = cmd_parsers.add_parser(name)
    topologies = KnownTopologies(cmd)
    collectives = KnownCollectives(cmd)
    validate_output_args, output_handler = add_output_msccl_objects(cmd)
    cmd.add_argument('--min-chunks', type=int, default=1, metavar='N')
    cmd.add_argument('--max-chunks', type=int, default=None, metavar='N')
    cmd.add_argument('--assume-rpc-bound', default=None, help='assume bandwidth optimality requires at least this many rounds per chunk', metavar='N/N')
    cmd.add_argument('--no-monotonic-feasibility', action='store_true', help='turn off an unproven assumption about monotonic feasibility of instances')
    cmd.add_argument('--save-eagerly', action='store_true', help='save algorithms as soon as they are found, without pruning non-Pareto optimal algorithms at the end')
    instance_handler = add_instance(cmd, take_steps=False, take_rounds=False)

    def handle(args, command):
        if command != name:
            return False

        validate_output_args(args)
        topology = topologies.create(args)
        instance = instance_handler(args)
        collective = collectives.create(args, topology.num_nodes())
        assume_rpc_bound = None
        if args.assume_rpc_bound:
            try:
                assume_rpc_bound = parse_fraction(args.assume_rpc_bound)
            except ValueError:
                cmd.error('could not parse --assume-rpc-bound as a fraction')
        algorithms = []
        for algorithm in strategies.solve_all_latency_bandwidth_tradeoffs(topology, collective, args.min_chunks, args.max_chunks, assume_rpc_bound, not args.no_monotonic_feasibility, base_instance=instance, logging=True):
            algorithms.append(algorithm)
            if args.save_eagerly:
                output_handler(args, algorithm, algorithm.name)
        if not args.save_eagerly:
            efficient_algorithms = strategies.prune_pareto_optimal(algorithms)
            print(f'Found {len(efficient_algorithms)} Pareto optimal algorithms. Pruned {len(algorithms) - len(efficient_algorithms)} non-optimal algorithms.')
            for algorithm in efficient_algorithms:
                output_handler(args, algorithm, algorithm.name)
        return True
    
    return handle
