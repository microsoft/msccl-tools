# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from dataclasses import dataclass
from sccl.language.ir import *

@dataclass
class Chunk:
    origin_rank: int # Rank the chunk initially started at
    origin_index: int # Index the chunk initially started at
    dst_rank: int = -1
    dst_index: int = -1

    def reduce(self, chunk):
        if type(chunk) is ReduceChunk:
            return chunk.reduce(self)
        elif type(chunk) is Chunk:  
            chunks = [self, chunk]
            return ReduceChunk(chunks)
        else:
            assert True, "Trying to reduce with chunk of None"
            return None

    def __hash__(self):
        return hash((self.origin_rank, self.origin_index))

    def __eq__(self, other):
        return type(other) is Chunk and self.origin_rank == other.origin_rank and self.origin_index == other.origin_index

    def __lt__(self, other):
        return self.origin_rank < other.origin_rank or \
               (self.origin_rank == other.origin_rank and self.origin_index < other.origin_index)

@dataclass
class ReduceChunk:
    chunks: list # List of chunks reduced

    def reduce(self, chunk):
        if type(chunk) is ReduceChunk:
            chunks = self.chunks + chunk.chunks
        elif type(chunk) is Chunk:  
            chunks =self.chunks + [chunk]
        else:
            assert True, "Trying to reduce with chunk of None"
        return ReduceChunk(chunks)

    def sort(self):
        self.chunks.sort()

    def __hash__(self):
        return hash(tuple(self.chunks))

    # Two reduce chunks are equal if they contain the same list of
    # chunks being reduced
    def __eq__(self, other):
        self.sort()
        other.sort()
        return self.chunks == other.chunks
