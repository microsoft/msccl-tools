# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from enum import Enum

_current_program = None
def _curr():
    global _current_program
    if _current_program == None:
        raise RuntimeError("No Program in context")
    return _current_program

class Buffer(Enum):
    Input = 0,
    Output = 1,
    Scratch = 2

class SCCLProgram:
    def __init__(self, name, topo):
        self.name = name
        self.topo = topo
        self.ranks = [Process(self, r) for r in range(topo.num_nodes())]
        self.ops = []

    def rank(self, rank):
        return self.ranks[rank]

    def _add_op(self, op):
        self.ops.append(op)

    def __enter__(self):
        global _current_program
        if _current_program != None:
            raise RuntimeError("There is already a SCCL Program in context")
        _current_program = self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _current_program
        if _current_program != self:
            raise RuntimeError("This program is not currently in context")
        _current_program = None

def Rank(index):
    return _curr().rank(index)

@dataclass
class Process:
    prog: SCCLProgram
    rank: int

    def input(self, id):
        # TODO: mark input
        return Ref(self.prog, self.rank, id)

@dataclass
class Ref:
    prog: SCCLProgram
    proc: Process
    id: int
    step: int = 0

    def _get_ref(self, dst):
        if isinstance(dst, Process):
            return Ref(self.prog, dst, self.id)
        elif isinstance(dst, Ref):
            return dst
        else:
            raise TypeError("Unsupported type")

    def send(self, dst):
        dst_ref = self._get_ref(dst)
        self.prog._add_op(TransferOp(OpCode.Send, self, dst_ref))
        return dst_ref
    
    def wait(self, steps):
        # TODO: fix this
        future = Ref(self.prog, self.proc, )
        self.prog._add_op(TransferOp(OpCode.Wait, self, future))
        return future

    def output(self):
        # TODO: mark output
        return self

    def reduce(self, other):
        # TODO: do something
        return self

class OpCode(Enum):
    Copy = 0
    Send = 1

@dataclass
class Op:
    op_code: OpCode

@dataclass
class TransferOp(Op):
    src_ref: Ref
    dst_ref: Ref = None
