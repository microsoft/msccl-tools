# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import igraph as ig
from msccl.language.ir import *
from msccl.language.rank_dag import *

def visualize_chunk_dag(chunk_paths): # pragma: no cover
    frontier = []
    nnodes = 0
    vertex_label = []
    vertex_colors = []
    edges = []
    visited = set()

    def add_node(op, nnodes, vertex_label, vertex_colors):
        if op.num == -1:
            op.num = nnodes
            nnodes += 1
            if op.inst == ChunkInstruction.start:
                vertex_label.append(f'Start at {op.dst.rank}, {op.dst.index}.')
                vertex_colors.append('yellow')
            elif op.inst == ChunkInstruction.send:
                vertex_label.append(f'Send to Rank {op.dst.rank} {op.dst.index}. {op.steps_to_end}, {op.steps_from_start}')
                vertex_colors.append('blue')
            elif op.inst == ChunkInstruction.reduce:
                vertex_label.append(f'Reduce with {op.dst.rank} {op.dst.index}. {op.steps_to_end}, {op.steps_from_start}')
                vertex_colors.append('green')
        return nnodes

    for chunk, op in chunk_paths.items():
        if len(op.prev) == 0: 
            frontier.append(op)

    while len(frontier) > 0:
        op = frontier[0]
        if op in visited:
            frontier = frontier[1:]
        else:
            nnodes = add_node(op, nnodes, vertex_label, vertex_colors)
            for next_op in op.next:
                nnodes = add_node(next_op, nnodes, vertex_label, vertex_colors)
                edges.append([op.num, next_op.num])
            frontier = frontier[1:] + op.next
            visited.add(op)

    g = ig.Graph(nnodes, edges, directed=True)
    layout = g.layout(layout=ig.Graph.layout_grid)
    ig.plot(g, vertex_label=vertex_label, vertex_color=vertex_colors, layout='auto')

def visualize_rank_dag(operations): # pragma: no cover
    frontier = []
    nnodes = 0
    vertex_label = []
    vertex_colors = []
    edges = []
    visited = set()
    colors = ['red', 'green', 'blue', 'yellow', 'teal', 'pink', 'purple', 'orange']

    def add_node(op, nnodes, vertex_label, vertex_colors):
        if op.num == -1:
            op.num = nnodes
            nnodes += 1
            # Add new node to graph
            if op.inst == Instruction.start:
                vertex_label.append(f'Chunk {op.src.index} Rank {op.src.rank}')
            elif op.inst == Instruction.send:
                vertex_label.append(f'S to Rank {op.dst.rank}')
            elif op.inst == Instruction.recv:
                vertex_label.append(f'R from {op.src.rank}')
            elif op.inst == Instruction.recv_reduce_copy:
                vertex_label.append(f'RRC from {op.src.rank}')
            else:
                vertex_label.append(f'{op.inst}')

            # Add colors 
            if op.inst == Instruction.start:
                vertex_colors.append('gray')
            else:
                vertex_colors.append(colors[op.tb % len(colors)])
        return nnodes

    for slot, op in operations.items():
        if len(op.prev) == 0: 
            frontier.append(op)

    while len(frontier) > 0:
        op = frontier[0]

        if op in visited:
            frontier = frontier[1:]
        else:
            nnodes = add_node(op, nnodes, vertex_label, vertex_colors)

        for next_op in op.next:
            nnodes = add_node(next_op, nnodes, vertex_label, vertex_colors)
            edges.append([op.num, next_op.num])
            frontier = frontier[1:] + list(op.next)
        visited.add(op)

    g = ig.Graph(nnodes, edges, directed=True)
    layout = g.layout(layout=ig.Graph.layout_grid)
    ig.plot(g, vertex_label=vertex_label, vertex_color=vertex_colors, layout='rt')