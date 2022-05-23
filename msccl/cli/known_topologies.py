# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import msccl.topologies as topologies
from msccl.serialization import *
from .known_transformers import KnownTransformers
from pathlib import Path
import sys

class KnownTopologies:
    def __init__(self, parser, tag=''):
        self.parser = parser
        self.tag = tag
        self.constructors = {
            'FullyConnected': self._sized_topo(topologies.fully_connected),
            'HubAndSpoke': self._sized_topo(topologies.hub_and_spoke),
            'Ring': self._sized_topo(topologies.ring),
            'Line': self._sized_topo(topologies.line),
            'Star': self._sized_topo(topologies.star),
            'AMD4': self._fixed_topo(topologies.amd4),
            'AMD8': self._fixed_topo(topologies.amd8),
            'DGX1': self._fixed_topo(topologies.dgx1),
            'DGX2': self._fixed_topo(lambda: topologies.hub_and_spoke(16)),
            'NVLinkOnly': self._fixed_topo(topologies.nvlink_only),
            'custom': self._custom_topo(),
        }
        self.parser.add_argument(f'topology{tag}', type=str, choices=self.constructors.keys(), help=f'topology {tag}')
        self.parser.add_argument(f'--topology-file{tag}', type=Path, default=None, help=f'a serialized topology', metavar=f'FILE')
        self.parser.add_argument(f'-n{tag}', f'--nodes{tag}', type=int, help='required for non-fixed topologies', metavar='N')
        self.known_transformers = KnownTransformers(parser, tag=tag)

    def _topology(self, args):
        return vars(args)[f'topology{self.tag}']

    def _nodes(self, args):
        return vars(args)[f'nodes{self.tag}']

    def create(self, args):
        topology = self.constructors[self._topology(args)](args)
        topology = self.known_transformers.transform(args, topology)
        return topology

    def _custom_topo(self):
        def make(args):
            input_file = vars(args)[f'topology_file{self.tag}']
            if input_file is None:
                self.parser.error(f'--topology-file{self.tag} is required for custom topologies')
                exit(1)

            if not input_file.exists():
                print(f'error: input file not found: {input_file}', file=sys.stderr)
                exit(1)

            return load_msccl_object(input_file)
        return make

    def _fixed_topo(self, Cls):
        def make(args):
            topo = Cls()
            if self._nodes(args) != None and self._nodes(args) != topo.num_nodes():
                self.parser.error(f'fixed-size topology {self._topology(args)} has {topo.num_nodes()} nodes, but command line specified {self._nodes(args)} nodes')
            return topo
        return make

    def _sized_topo(self, Cls):
        def make(args):
            if self._nodes(args) == None:
                self.parser.error(f'topology {self._topology(args)} requires -n/--nodes')
            return Cls(self._nodes(args))
        return make
