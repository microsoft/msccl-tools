#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sccl.collectives as collectives
import sccl.topologies as topologies
import sccl.strategies as strategies
from sccl.cli import *

import argparse
import argcomplete
import sys

def main():
    parser = argparse.ArgumentParser('sccl')

    cmd_parsers = parser.add_subparsers(title='command', dest='command')
    cmd_parsers.required = True

    handlers = []
    handlers.append(make_solvers(cmd_parsers))
    handlers.append(make_distributors(cmd_parsers))
    handlers.append(make_analyses(cmd_parsers))
    handlers.append(make_handle_ncclize(cmd_parsers))
    handlers.append(make_plans(cmd_parsers))

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    for handler in handlers:
        if handler(args, args.command):
            break

if __name__ == '__main__':
    main()
