# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .common import *
from msccl.autosynth import *

def make_plans(cmd_parsers):
    handler_funcs = []
    handler_funcs.append(make_handle_list)

    return make_cmd_category(cmd_parsers, 'plans', 'subcommand', handler_funcs)

def make_handle_list(cmd_parsers):
    cmd = cmd_parsers.add_parser('list')

    def handle(args, command):
        if command != 'list':
            return False

        print_plans()
        return True
    
    return handle
