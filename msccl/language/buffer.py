# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Scratch buffer slice with manual indexing
from msccl.language.ir import Buffer


class BufferSlice:
    def __init__(self, buf: Buffer, name: str):
        self.name = name
        self.buf = buf
        self.offset: int = -1 # Offset into the global scratch buffer
        self.chunks: list = []

    # Returns the global index into the scratch buffer
    def get_global_index(self, index: int):
        assert (self.offset > -1), 'set_offset needs to be called first'
        return self.offset + index

    def get_buffer(self):
        return self.buf

    def instance_size(self):
        return len(self.chunks)

    def set_offset(self, offset):
        self.offset = offset

    def __getitem__(self, index):
        return self.chunks[index]
    
    def __setitem__(self, index, value):
        current_size = len(self.chunks)
        while index > current_size:
            self.chunks.append(None)
            current_size = len(self.chunks)
        if index == current_size:
            self.chunks.append(value)
        else:
            self.chunks[index] = value

    def __len__(self):
        return len(self.chunks)