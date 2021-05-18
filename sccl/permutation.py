# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass

@dataclass
class Permutation:
    nodes: list
    chunks: list = None

    def chunk_up(self, div):
        if self.chunks is None:
            return self

        chunks = []
        for chunk, perm in enumerate(self.chunks):
            for i in range(div):
                chunks.append(perm * div + i)

        return Permutation(self.nodes, chunks)

    def __str__(self):
        s = f'Permutation(nodes={self.nodes}'
        if self.chunks != None:
            s += f', chunks={self.chunks}'
        return s + ')'
