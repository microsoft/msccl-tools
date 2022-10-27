# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.composers import *
from .common import *

def make_composers(cmd_parsers):
    handler_funcs = []
    handler_funcs.append(make_handle_allreduce)

    return make_cmd_category(cmd_parsers, 'compose', 'composer', handler_funcs)

def make_handle_allreduce(cmd_parsers):
    name = 'allreduce'
    cmd = cmd_parsers.add_parser(name)
    read_reducescatter_algorithm = add_input_algorithm(cmd, name="reducescatter-algorithm")
    read_allgather_algorithm = add_input_algorithm(cmd, name="allgather-algorithm")
    validate_output_args, output_handler = add_output_algorithm(cmd)

    def handle(args, command):
        if command != name:
            return False

        reducescatter_algorithm = read_reducescatter_algorithm(args)
        allgather_algorithm = read_allgather_algorithm(args)
        validate_output_args(args)
        algo = compose_allreduce(reducescatter_algorithm, allgather_algorithm, logging=True)
        output_handler(args, algo)
        return True

    return handle