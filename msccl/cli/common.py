# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.serialization import *
from msccl.instance import *
from pathlib import Path
import sys
import re
from fractions import Fraction

def _legalize_msccl_name(name):
    name = name.replace('(', '.')
    name = name.replace('=', '')
    name = name.replace(',', '.')
    name = name.replace(')', '')
    return name

def name_msccl_object(name, ending='msccl.json'):
    return f'{_legalize_msccl_name(name)}.{ending}'

def _validate_output_directory(directory):
    if not directory.exists():
        print('error: output directory does not exists', file=sys.stderr)
        exit(1)
    if not directory.is_dir():
        print('error: output path is not a directory', file=sys.stderr)
        exit(1)

def _handle_write_to_directory(directory, force, get_contents, preferred_file_name):
    output_file = directory / preferred_file_name
    if output_file.exists():
        if output_file.is_dir():
            print(f'error: output path is a directory', file=sys.stderr)
            exit(1)
        if force:
            print(f'Overwriting {output_file}')
        else:
            print(f'file already exists, use -f/--force to overwrite {output_file}', file=sys.stderr)
            return False
    with output_file.open('w') as f:
        f.write(get_contents())
    print(f'Wrote to {output_file}')
    return True

def add_output_file(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-o', '--output', type=Path, help='file to write synthesized algorithm to', metavar='FILE')
    group.add_argument('-d', '--directory', type=Path, default=Path(), help='directory to write the synthesized algorithm to', metavar='DIR')
    parser.add_argument('-f', '--force', action='store_true', help='overwrite existing files')
    parser.add_argument('--no-save', action='store_true', help='do not save to file')

    def validate_args(args):
        if args.output != None:
            if args.output.is_dir():
                print(f'error: output path is a directory, did you mean to use -d?', file=sys.stderr)
                exit(1)
        if args.directory != None:
            _validate_output_directory(args.directory)

    def handle(args, get_contents, preferred_file_name):
        if args.no_save:
            return False
        if args.output != None:
            if args.output.exists() and not args.force:
                print(f'file already exists, use -f/--force to overwrite {args.output}', file=sys.stderr)
                return False
            with args.output.open('w') as f:
                f.write(get_contents())
            print(f'Wrote to {args.output}')
        else:
            return _handle_write_to_directory(args.directory, args.force, get_contents, preferred_file_name)
        return True

    return validate_args, handle

def add_output_algorithm(parser):
    validate_args, handle_file = add_output_file(parser)

    def handle(args, algorithm):
        if algorithm == None:
            return # Strategies/distributors have their specific failure prints

        handled = handle_file(args, lambda: MSCCLEncoder().encode(algorithm), name_msccl_object(algorithm.name))
        if not handled:
            print(f'\n{algorithm.name} algorithm:')
            print(algorithm)

    return validate_args, handle

def add_output_topology(parser):
    validate_args, handle_file = add_output_file(parser)

    def handle(args, topology):
        handled = handle_file(args, lambda: MSCCLEncoder().encode(topology), name_msccl_object(topology.name))

    return validate_args, handle

def add_output_msccl_objects(parser):
    parser.add_argument('-d', '--directory', type=Path, default=Path(), help='directory to write outputs to', metavar='DIR')
    parser.add_argument('-f', '--force', action='store_true', help='overwrite existing files')
    parser.add_argument('--no-save', action='store_true', help='do not save to file')

    def validate_args(args):
        _validate_output_directory(args.directory)

    def handle(args, msccl_object, name):
        if not args.no_save:
            _handle_write_to_directory(args.directory, args.force, lambda: MSCCLEncoder().encode(msccl_object), name_msccl_object(name))
    
    return validate_args, handle

def add_input_algorithm(parser, multiple=False, name='algorithm'):
    parser.add_argument(name, type=Path, nargs='+' if multiple else 1, help=f'algorithm to operate on')

    def read_algorithm(args):
        algos = []
        for input_file in vars(args)[name]:
            if not input_file.exists():
                print(f'error: input file not found: {input_file}', file=sys.stderr)
                exit(1)

            algo = load_msccl_object(input_file)
            algos.append(algo)
        if multiple:
            return algos
        else:
            return algos[0]

    return read_algorithm

def add_instance(parser, take_steps=True, take_rounds=True, take_chunks=True):
    if take_steps:
        parser.add_argument('-s', '--steps', type=int, required=True)
    if take_rounds:
        parser.add_argument('-r', '--rounds', type=int, default=None, metavar='N')
    if take_chunks:
        parser.add_argument('-c', '--chunks', type=int, default=1, metavar='N')
    parser.add_argument('--pipeline', type=int, default=None, metavar='N')
    parser.add_argument('--extra-memory', type=int, default=None, metavar='N')
    parser.add_argument('--allow-exchange', action='store_true')

    def handle(args):
        if take_rounds:
            if args.rounds != None:
                if args.rounds < args.steps:
                    parser.error(f'error: rounds cannot be less than steps ({args.rounds} < {args.steps})')
                extra_rounds = args.rounds - args.steps
            else:
                extra_rounds = 0
        return Instance(
            steps=args.steps if take_steps else None,
            extra_rounds=extra_rounds if take_rounds else 0,
            chunks=args.chunks if take_chunks else 1,
            pipeline=args.pipeline,
            extra_memory=args.extra_memory,
            allow_exchange=args.allow_exchange)

    return handle

def parse_fraction(value):
    try:
        return int(value)
    except ValueError:
        m = re.fullmatch('(.+)/(.+)', value)
        if m == None:
            raise ValueError('value must be in format "<numerator>/<denominator>"')
        numerator = int(m.group(1))
        denominator = int(m.group(2))
        return Fraction(numerator, denominator)

def make_cmd_category(cmd_parsers, name, title, handler_funcs):
    cmd = cmd_parsers.add_parser(name)
    category_parsers = cmd.add_subparsers(title=title, dest=title)
    category_parsers.required = True
    
    handlers = []
    for func in handler_funcs:
        handlers.append(func(category_parsers))
    
    def handle(args, command):
        if command != name:
            return False
        
        for handler in handlers:
            if handler(args, vars(args)[title]):
                return True
    
    return handle
