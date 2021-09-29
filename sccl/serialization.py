# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.algorithm import Algorithm, Step
from sccl.topologies import Topology
from sccl.instance import Instance
from sccl.collectives import Collective, Chunk

import json
import warnings

def _sccl_object_hook(o):
    if not 'sccl_type' in o:
        return o
    if o['sccl_type'] == 'algorithm':
        input_map = { int(k): set(v) for k, v in o['input_map'].items() }
        output_map = { int(k): set(v) for k, v in o['output_map'].items() }
        return Algorithm(o['name'], o['collective'], o['topology'], o['instance'], o['steps'], input_map, output_map)
    if o['sccl_type'] == 'step':
        sends = [(addr, src, dst) for addr, src, dst in o['sends']]
        return Step(o['rounds'], sends)
    if o['sccl_type'] == 'collective':
        triggers = { (int(r), int(c)): v for r, rmap in o['triggers'].items() for c, v in rmap.items() }
        return Collective(o['name'], o['nodes'], o['chunks'], triggers, o['runtime_name'])
    if o['sccl_type'] == 'chunk':
        pre = set(o['pre'])
        post = set(o['post'])
        return Chunk(pre, post, o['addr'])
    if o['sccl_type'] == 'topology':
        return Topology(o['name'], o['links'], o['switches'])
    if o['sccl_type'] == 'instance':
        return Instance(o['steps'], o['extra_rounds'], o['chunks'], o['pipeline'], o['extra_memory'], o['allow_exchange'])
    warnings.warn('Unhandled sccl_type in JSON')

def SCCLDecoder():
    return json.JSONDecoder(object_hook=_sccl_object_hook)

class SCCLEncoder(json.JSONEncoder):
    def __init__(self):
        super().__init__()
    
    def default(self, o):
        if isinstance(o, Algorithm):
            input_map = { k: list(v) for k, v in o.input_map.items() }
            output_map = { k: list(v) for k, v in o.output_map.items() }
            return {
                'sccl_type': 'algorithm',
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
                'sccl_type': 'step',
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
                'sccl_type': 'collective',
                'name': o.name,
                'nodes': o.num_nodes,
                'chunks': o._chunks,
                'triggers': triggers,
                'runtime_name': o.runtime_name,
            }
        if isinstance(o, Chunk):
            return {
                'sccl_type': 'chunk',
                'pre': list(o.precondition),
                'post': list(o.postcondition),
                'addr': o.address,
            }
        if isinstance(o, Topology):
            return {
                'sccl_type': 'topology',
                'name': o.name,
                'switches': o.switches,
                'links': o.links,
            }
        if isinstance(o, Instance):
            return {
                'sccl_type': 'instance',
                'steps': o.steps,
                'extra_rounds': o.extra_rounds,
                'chunks': o.chunks,
                'pipeline': o.pipeline,
                'extra_memory': o.extra_memory,
                'allow_exchange': o.allow_exchange,
            }
        return json.JSONEncoder.default(self, o)

def save_sccl_object(obj, filename):
    with open(filename, 'w') as f:
        f.write(SCCLEncoder().encode(obj))

def load_sccl_object(filename):
    with open(filename) as f:
        return SCCLDecoder().decode(f.read())
