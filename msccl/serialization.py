# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.algorithm import Algorithm, Step
from msccl.topologies import Topology
from msccl.instance import Instance
from msccl.collectives import Collective, Chunk

import json
import warnings

def _msccl_object_hook(o):
    if not 'msccl_type' in o:
        return o
    if o['msccl_type'] == 'algorithm':
        input_map = { int(k): set(v) for k, v in o['input_map'].items() }
        output_map = { int(k): set(v) for k, v in o['output_map'].items() }
        return Algorithm(o['name'], o['collective'], o['topology'], o['instance'], o['steps'], input_map, output_map)
    if o['msccl_type'] == 'step':
        sends = [(addr, src, dst) for addr, src, dst in o['sends']]
        return Step(o['rounds'], sends)
    if o['msccl_type'] == 'collective':
        triggers = { (int(r), int(c)): v for r, rmap in o['triggers'].items() for c, v in rmap.items() }
        return Collective(o['name'], o['nodes'], o['chunks'], triggers, o['runtime_name'])
    if o['msccl_type'] == 'chunk':
        pre = set(o['pre'])
        post = set(o['post'])
        return Chunk(pre, post, o['addr'])
    if o['msccl_type'] == 'topology':
        return Topology(o['name'], o['links'], o['switches'])
    if o['msccl_type'] == 'instance':
        return Instance(o['steps'], o['extra_rounds'], o['chunks'], o['pipeline'], o['extra_memory'], o['allow_exchange'])
    warnings.warn('Unhandled msccl_type in JSON')

def MSCCLDecoder():
    return json.JSONDecoder(object_hook=_msccl_object_hook)

class MSCCLEncoder(json.JSONEncoder):
    def __init__(self):
        super().__init__()
    
    def default(self, o):
        if isinstance(o, Algorithm):
            input_map = { k: list(v) for k, v in o.input_map.items() }
            output_map = { k: list(v) for k, v in o.output_map.items() }
            return {
                'msccl_type': 'algorithm',
                'name': o.name,
                'instance': o.instance,
                'input_map': input_map,
                'output_map': output_map,
                'steps': o.steps,
                'collective': o.collective,
                'topology': o.topology,
            }
        if isinstance(o, Step):
            return {
                'msccl_type': 'step',
                'rounds': o.rounds,
                'sends': o.sends,
            }
        if isinstance(o, Collective):
            triggers = {}
            for (r, c), v in o._triggers.items():
                if not r in triggers:
                    triggers[r] = {}
                triggers[r][c] = v
            return {
                'msccl_type': 'collective',
                'name': o.name,
                'nodes': o.num_nodes,
                'chunks': o._chunks,
                'triggers': triggers,
                'runtime_name': o.runtime_name,
            }
        if isinstance(o, Chunk):
            return {
                'msccl_type': 'chunk',
                'pre': list(o.precondition),
                'post': list(o.postcondition),
                'addr': o.address,
            }
        if isinstance(o, Topology):
            return {
                'msccl_type': 'topology',
                'name': o.name,
                'switches': o.switches,
                'links': o.links,
            }
        if isinstance(o, Instance):
            return {
                'msccl_type': 'instance',
                'steps': o.steps,
                'extra_rounds': o.extra_rounds,
                'chunks': o.chunks,
                'pipeline': o.pipeline,
                'extra_memory': o.extra_memory,
                'allow_exchange': o.allow_exchange,
            }
        return json.JSONEncoder.default(self, o)

def save_msccl_object(obj, filename):
    with open(filename, 'w') as f:
        f.write(MSCCLEncoder().encode(obj))

def load_msccl_object(filename):
    with open(filename) as f:
        return MSCCLDecoder().decode(f.read())
