import msccl.collectives as collectives
from enum import Enum

_current_program = None


def _curr():
    global _current_program
    if _current_program == None:
        raise RuntimeError('No Program in context')
    return _current_program


class MSCCLProgramV2:
    def __init__(self, name, collective):
        self.name = name
        self.coll = collective
        self.body = Body()
        self.stack = [self]
        self.buffers = {}

    def __enter__(self):
        global _current_program
        if _current_program != None:
            raise RuntimeError('There is already a MSCCL Program in context')
        _current_program = self
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _current_program
        if _current_program != self:
            raise RuntimeError('This program is not currently in context')
        _current_program = None

    def _curr_body(self):
        return self.stack[-1].body

    def get_buffer(self, name):
        if name not in self.buffers:
            self.buffers[name] = Buffer(name)
        return self.buffers[name]

    def add_stmt(self, stmt):
        self._curr_body().stmts.append(stmt)

    def add_opt_rule(self, rule):
        self.opt_rules.append(rule)

    def __repr__(self):
        return f'MSCCLProgramV2({self.name}, {self.coll.name}):\n{self.body}'


class Buffer:
    def __init__(self, name):
        self.name = name

    def __getitem__(self, indices):
        rank, chunk = indices
        return ChunkExpr(ChunkExpr.Kind.CHUNK, self.name, rank, chunk)

    def __setitem__(self, indices, value):
        raise RuntimeError('Cannot assign to buffer')

    def __repr__(self):
        return f'Buffer({self.name})'


class Body:
    def __init__(self):
        self.stmts = []

    def __repr__(self):
        lines = '\n'.join([str(stmt) for stmt in self.stmts])
        indented = '\n'.join([f'  {line}' for line in lines.splitlines()])
        return indented


class ForRangeStmt:
    def __init__(self, for_decl, var_name):
        self.decl = for_decl
        self.var_name = var_name
        self.body = Body()

    def __repr__(self):
        return f'{self.decl} as {self.var_name}:\n{self.body}'


def _id_to_var_name(id):
    '''Converts numeric id to index name, starting from i,j,k and defaulting to idx_id when necessary.'''
    if id < 15:
        return chr(ord('i') + id)
    else:
        return f'idx_{id}'


class ForRangeDecl:
    def __init__(self, start, end, step):
        self.domain = (start, end, step)

    def __enter__(self):
        var_name = _id_to_var_name(len(_curr().stack))
        stmt = ForRangeStmt(self, var_name)
        _curr().add_stmt(stmt)
        _curr().stack.append(stmt)
        return IndexExpr(IndexExpr.Kind.FOR_RANGE_VAR, var_name)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        assert len(_curr().stack) > 0, 'Exiting a loop when not in one'
        popped = _curr().stack.pop()
        assert isinstance(
            popped, ForRangeStmt), 'Expected to exit a for-range loop'
        assert popped.decl == self, 'Incorrect loop nesting'

    def __repr__(self):
        return f'for_range{self.domain}'


def for_range(start_or_end, end=None, step=1):
    if end == None:
        return ForRangeDecl(0, start_or_end, step)
    else:
        return ForRangeDecl(start_or_end, end, step)


class ConditionStmt:
    def __init__(self, decl):
        self.decl = decl
        self.body = Body()

    def __repr__(self):
        return f'if {self.decl}:\n{self.body}'


class ConditionDecl:
    def __init__(self, cond):
        self.cond = cond

    def __enter__(self):
        stmt = ConditionStmt(self)
        _curr().add_stmt(stmt)
        _curr().stack.append(stmt)
        return stmt

    def __exit__(self, exc_type, exc_value, exc_traceback):
        assert len(_curr().stack) > 0, 'Exiting a condition when not in one'
        popped = _curr().stack.pop()
        assert isinstance(
            popped, ConditionStmt), 'Expected to exit a condition'
        assert popped.decl == self, 'Incorrect condition nesting'

    def __repr__(self):
        return str(self.cond)


def condition(cond):
    return ConditionDecl(cond)


class _Bufs:
    def __getattribute__(self, name):
        return _curr().get_buffer(name)

    def __getitem__(self, name):
        return _curr().get_buffer(name)


Buf = _Bufs()


class ChunkExpr:
    class Kind(Enum):
        CHUNK = 'chunk'
        COPY = 'copy'
        REDUCE = 'reduce'

    def __init__(self, kind, *args):
        self.kind = kind
        self.args = args

    def copy(self, dst):
        _curr().add_stmt(ChunkExpr(ChunkExpr.Kind.COPY, self, dst))
        return dst

    def reduce(self, src):
        _curr().add_stmt(ChunkExpr(ChunkExpr.Kind.REDUCE, self, src))
        return self

    def __repr__(self):
        if self.kind == ChunkExpr.Kind.CHUNK:
            return f'{self.args[0]}[{self.args[1]},{self.args[2]}]'
        elif self.kind == ChunkExpr.Kind.COPY:
            return f'{self.args[0]}.copy({self.args[1]})'
        elif self.kind == ChunkExpr.Kind.REDUCE:
            return f'{self.args[0]}.reduce({self.args[1]})'
        else:
            raise RuntimeError(f'Unknown ChunkExpr kind {self.kind}')


class IndexExpr:
    class Kind(Enum):
        FOR_RANGE_VAR = 'for_range_var'
        PARAM = 'param'
        ADD = 'add'
        SUB = 'sub'
        MUL = 'mul'
        MOD = 'mod'
        EQ = 'eq'
        NE = 'ne'

    def __init__(self, kind, *args):
        self.kind = kind
        self.args = args

    def __add__(self, other):
        return IndexExpr(IndexExpr.Kind.ADD, self, other)

    def __radd__(self, other):
        return IndexExpr(IndexExpr.Kind.ADD, other, self)

    def __sub__(self, other):
        return IndexExpr(IndexExpr.Kind.SUB, self, other)

    def __rsub__(self, other):
        return IndexExpr(IndexExpr.Kind.SUB, other, self)

    def __mul__(self, other):
        return IndexExpr(IndexExpr.Kind.MUL, self, other)

    def __rmul__(self, other):
        return IndexExpr(IndexExpr.Kind.MUL, other, self)

    def __mod__(self, other):
        return IndexExpr(IndexExpr.Kind.MOD, self, other)

    def __rmod__(self, other):
        return IndexExpr(IndexExpr.Kind.MOD, other, self)

    def __eq__(self, other):
        return IndexExpr(IndexExpr.Kind.EQ, self, other)

    def __ne__(self, other):
        return IndexExpr(IndexExpr.Kind.NE, self, other)

    def __repr__(self):
        if self.kind == IndexExpr.Kind.FOR_RANGE_VAR:
            return self.args[0]
        elif self.kind == IndexExpr.Kind.PARAM:
            return self.args[0]
        elif self.kind == IndexExpr.Kind.ADD:
            return f'({self.args[0]}+{self.args[1]})'
        elif self.kind == IndexExpr.Kind.SUB:
            return f'({self.args[0]}-{self.args[1]})'
        elif self.kind == IndexExpr.Kind.MUL:
            return f'({self.args[0]}*{self.args[1]})'
        elif self.kind == IndexExpr.Kind.MOD:
            return f'({self.args[0]}%{self.args[1]})'
        elif self.kind == IndexExpr.Kind.EQ:
            return f'({self.args[0]}=={self.args[1]})'
        elif self.kind == IndexExpr.Kind.NE:
            return f'({self.args[0]}!={self.args[1]})'
        else:
            raise RuntimeError(f'Unknown IndexExpr kind: {self.kind}')


def allreduce_ring(size, channels):
    # Reduce ring
    with for_range(size-1) as step:
        with for_range(size) as index:
            rank = (index + step) % size
            next_rank = (index + step + 1) % size
            Buf.input[next_rank, index].reduce(Buf.input[rank, index])

    # Propagate ring
    with for_range(-1, size-2) as step:
        with for_range(size) as index:
            rank = (index + step) % size
            next_rank = (index + step + 1) % size
            Buf.input[rank, index].copy(Buf.input[next_rank, index])


def allreduce_allpairs(size):
    # Each rank sends the nth chunk to the nth rank into scratch space
    with for_range(size) as r1:
        with for_range(size) as r2:
            with condition(r1 != r2):
                index = r2 * size
                Buf.input[r1, index].copy(Buf.scratch[r2, index])

    # Each rank performs a local reduction on the nth chunk
    # Utilize 8 threadblocks for this reduction for better parallelism
    with for_range(size) as r:
        with for_range(0, size * (size-1)) as index:
            Buf.input[r, r*size + (index % size)].reduce(Buf.scratch[r, index])

    # Each rank sends the fully reduced nth chunk to all other gpus
    with for_range(size) as r1:
        with for_range(size) as r2:
            with condition(r1 != r2):
                index = r1 * size
                Buf.input[r1, index].copy(Buf.input[r2, index])


if __name__ == '__main__':
    with MSCCLProgramV2('allreduce_ring', collectives.allreduce(4).chunk_up(4)) as prog:
        allreduce_ring(4, 1)
        print(prog)
    print('------------------------------------------------------------')
    with MSCCLProgramV2('allreduce_allpairs', collectives.allreduce(4).chunk_up(4)) as prog:
        allreduce_allpairs(4)
        print(prog)
