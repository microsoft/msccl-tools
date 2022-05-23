# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import msccl.collectives as collectives
from msccl.serialization import *
from pathlib import Path
import sys

class KnownCollectives:
    def __init__(self, parser):
        self.parser = parser
        self.constructors = {
            'Broadcast': self._rooted_coll(collectives.broadcast),
            'Reduce': self._rooted_coll(collectives.reduce),
            'Scatter': self._rooted_coll(collectives.scatter),
            'Gather': self._rooted_coll(collectives.gather),
            'Allgather': self._coll(collectives.allgather),
            'Allreduce': self._coll(collectives.allreduce),
            'Alltoall': self._coll(collectives.alltoall),
            'ReduceScatter': self._coll(collectives.reduce_scatter),
            'Scan': self._coll(collectives.scan),
            'MultirootBroadcast': self._multiroot_coll(collectives.multiroot_broadcast),
            'MultirootScatter': self._multiroot_coll(collectives.multiroot_scatter),
            'MultirootGather': self._multiroot_coll(collectives.multiroot_gather),
            'custom': self._custom_coll(),
        }
        self.parser.add_argument('collective', type=str, choices=self.constructors.keys(), help='collective')
        self.parser.add_argument('--collective-file', type=Path, default=None, help='a serialized collective', metavar='FILE')
        self.parser.add_argument('--root', type=int, default=0, help='used by rooted collectives', metavar='N')
        self.parser.add_argument('--roots', type=int, nargs='+', default=[0], help='used by multi-rooted collectives', metavar='N')

    def create(self, args, num_nodes):
        return self.constructors[args.collective](num_nodes, args)

    def _custom_coll(self):
        def make(size, args):
            input_file = args.collective_file
            if input_file is None:
                self.parser.error('--collective-file is required for custom collectives')
                exit(1)

            if not input_file.exists():
                print(f'error: input file not found: {input_file}', file=sys.stderr)
                exit(1)

            return load_msccl_object(input_file)
        return make

    def _rooted_coll(self, fun):
        def make(size, args):
            root = args.root
            return fun(size, root)
        return make

    def _coll(self, fun):
        def make(size, args):
            return fun(size)
        return make

    def _multiroot_coll(self, fun):
        def make(size, args):
            roots = args.roots
            return fun(size, roots)
        return make
