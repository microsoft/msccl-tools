# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import msccl.topologies as topologies

class KnownTransformers:
    def __init__(self, parser, tag=''):
        self.parser = parser
        self.tag = tag
        self.transformers = {
            'reverse': topologies.reverse_topology,
            'binarize': topologies.binarize_topology,
        }
        self.parser.add_argument(f'-t{tag}', f'--transform{tag}', action='append', default=[], choices=self.transformers.keys(), help='apply a topology transformer. may be used multiple times')

    def transform(self, args, topology):
        for key in vars(args)[f'transform{self.tag}']:
            topology = self.transformers[key](topology)
        return topology
