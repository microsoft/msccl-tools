# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass

@dataclass(frozen=True)
class Instance:
    steps: int
    extra_rounds: int = 0
    chunks: int = 1
    pipeline: int = None
    extra_memory: int = None
    allow_exchange: bool = False

    def rounds(self):
        return self.steps + self.extra_rounds

    def set(self, steps = None, extra_rounds = None, chunks = None, pipeline = None, extra_memory = None, allow_exchange = None):
        return Instance(
            steps if steps != None else self.steps,
            extra_rounds if extra_rounds != None else self.extra_rounds,
            chunks if chunks != None else self.chunks,
            pipeline if pipeline != None else self.pipeline,
            extra_memory if extra_memory != None else self.extra_memory,
            allow_exchange if allow_exchange != None else self.allow_exchange)
    
    def __str__(self):
        s = f'steps={self.steps}'
        if self.extra_rounds > 0:
            s += f',rounds={self.steps + self.extra_rounds}'
        if self.chunks > 1:
            s += f',chunks={self.chunks}'
        if self.pipeline != None:
            s += f',pipeline={self.pipeline}'
        if self.extra_memory != None:
            s += f',extra_memory={self.extra_memory}'
        if self.allow_exchange:
            s += f',allow_exchange'
        return s
