# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from lxml import etree as ET
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


@dataclass
class Program:
    name: str
    collective: str
    inplace: bool
    protocol: str
    gpus: list = field(default_factory=list)


@dataclass
class Gpu:
    rank: int
    threadblocks: list = field(default_factory=list)

    # From ncclize
    precopies: list = field(default_factory=list)
    postcopies: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    input_chunks: int = 0
    output_chunks: int = 0
    scratch_chunks: int = 0
    scratch: dict = field(default_factory=dict)
    channels: dict = field(default_factory=dict)

    def scratch_size(self):
        return max((idx for addr, idx in self.scratch.items()), default=-1) + 1

class ChannelType(Enum):
    proxy = 'proxy'
    sm = 'sm'
    none = 'none'

    def __str__(self):
        return self.value

class Buffer(Enum):
    input = 'i'
    output = 'o'
    scratch = 's'

    def __str__(self):
        return self.value

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value < other.value

@dataclass(frozen=True)
class Channel:
    srcBuffer: Buffer
    dstBuffer: Buffer
    type: ChannelType
    connected_to: int


@dataclass
class Threadblock:
    id: int = -1
    channel: int = -1
    send: int = -1
    recv: int = -1
    ops: list = field(default_factory=list)
    rbid: int = -1 # threadblock id of the receiver
    channels: list = field(default_factory=list)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class ChunkInstruction(Enum):
    start = 'start'
    reduce = 'reduce'
    send = 'send'

    def __str__(self):
        return self.value


class ThreadblockPolicy(Enum):
    auto = 'auto'
    manual = 'manual'

    def __str__(self):
        return self.value


class Instruction(Enum):
    nop = 'nop'
    send = 's'
    recv = 'r'
    recv_copy_send = 'rcs'
    recv_reduce_send = 'rrs'
    recv_reduce_copy = 'rrc'
    recv_reduce_copy_send = 'rrcs'
    read_reduce_copy = "rrc"
    read_reduce_copy_send = "rrcs"
    reduce_send = 'rs'
    copy = 'cpy'
    reduce = 're'
    delete = 'd'
    start = 'st'
    put = 'put'
    get = 'get'
    wait = 'wait'
    signal = 'signal'
    flush = 'flush'

    def __str__(self):
        return self.value



@dataclass
class ChunkRef:
    rank: int
    buffer: Buffer
    index: int
    size: int

    def __hash__(self):
        return hash((self.rank, self.buffer, self.index, self.size))


@dataclass
class Op:
    inst: Instruction
    rank: int
    src: ChunkRef
    dst: ChunkRef
    depends: list = field(default_factory=list)
    step: int = -1 # Step in the TB
    tb: int = -1 # TB this op is assigned to
    prev: list = field(default_factory=list) # List of instructions that happen before
    next: list = field(default_factory=list) # List of instructions that happen after
    num: int = -1
    chunk_step: int = -1
    priority: int = -1
    recv_match =  None
    send_match =  None
    channel: int = -1
    channel_type: ChannelType = ChannelType.none
    srcs: list = field(default_factory=list)
    dsts: list = field(default_factory=list)

    def cnt(self):
        if self.src:
            if self.dst:
                assert self.src.size == self.dst.size
            return self.src.size
        elif self.dst:
            return self.dst.size
        else:
            return 0

    def is_send(self):
         return self.inst == Instruction.send or \
            self.inst == Instruction.recv_reduce_copy_send or \
            self.inst == Instruction.recv_copy_send or \
            self.inst == Instruction.recv_reduce_send

    def is_recv(self):
        return  self.inst == Instruction.recv or \
            self.inst == Instruction.recv_reduce_copy or \
            self.inst == Instruction.recv_reduce_copy_send or \
            self.inst == Instruction.recv_copy_send or \
            self.inst == Instruction.recv_reduce_send

    def is_fused(self):
        return self.inst == Instruction.recv_reduce_copy_send or \
            self.inst == Instruction.recv_copy_send or \
            self.inst == Instruction.recv_reduce_send

    def is_local(self):
        return self.inst == Instruction.copy or \
            self.inst == Instruction.reduce

    def peer(self):
        if self.inst == Instruction.send:
            return self.dst.rank
        elif self.inst == Instruction.recv:
            return self.src.rank
        else:
            return None

    def send_peer(self):
        if self.is_send():
            return self.dst.rank
        return -1

    def recv_peer(self):
        if self.is_recv():
            return self.src.rank
        return -1

    def __eq__(self, other):
        return self is other

    def __lt__(self, other):
        # Ordering of operations
        # 1. Lower chunk step 2. Higher priority 3. Lower src index
        if self.chunk_step == other.chunk_step:
            if self.priority == other.priority:
                return self.src.index < other.src.index
            return self.priority > other.priority
        return self.chunk_step < other.chunk_step

    def __gt__(self, other):
        return not self < other

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f'Op({self.inst}, {self.rank}, {self.src}, {self.dst}, step:{self.step}, tb:{self.tb})'


# Instructions where src is on local GPU
_local_src_insts = {Instruction.send, Instruction.copy, Instruction.reduce}
# Instructions where dst is on local GPU
_local_dst_insts = {Instruction.recv, Instruction.recv_copy_send, Instruction.recv_reduce_send,
                    Instruction.recv_reduce_copy, Instruction.copy, Instruction.reduce,
                    Instruction.recv_reduce_copy_send}

_local_src_insts_mscclpp = {Instruction.put, Instruction.signal, Instruction.copy, Instruction.reduce, Instruction.reduce_send}
_local_dst_insts_mscclpp = {Instruction.get, Instruction.wait, Instruction.read_reduce_copy, Instruction.copy, Instruction.reduce, Instruction.read_reduce_copy_send, Instruction.reduce_send}


def ir_to_xml(program: Program, old_format=True, use_scratch=True, pretty_print=True, dependence_nop=False):
    # Figure out sizes of buffers based on usage
    buffer_sizes = defaultdict(lambda: 0)
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            for op in tb.ops:
                if op.inst in _local_src_insts:
                    key = (gpu.rank, op.src.buffer)
                    buffer_sizes[key] = max(
                        buffer_sizes[key], op.src.index + op.src.size)
                if op.inst in _local_dst_insts:
                    key = (gpu.rank, op.dst.buffer)
                    buffer_sizes[key] = max(
                        buffer_sizes[key], op.dst.index + op.dst.size)

    tb_id = {}
    # Sort threadblocks in each GPU by peers and then the channel
    # This is important as in NCCL threadblocks using the same NVLink concurrently should be close together
    for gpu in program.gpus:
        gpu.threadblocks = sorted(gpu.threadblocks,
                                  key=lambda tb: (tb.send, tb.recv, tb.channel))
        for i, tb in enumerate(gpu.threadblocks):
            tb_id[tb] = i

    # Filter out dependencies within the same threadblock
    op_tb_id = {}
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            for op in tb.ops:
                op_tb_id[op] = tb_id[tb]
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            for op in tb.ops:
                op.depends = list(
                    filter(lambda dep: op_tb_id[dep] != tb_id[tb], op.depends))
    # Filter out redundant dependencies
    # e.g. if op1 and op2 depend on op, and op1 happends before op2
    # then op2 does not need to explicitly depend on op
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            running_depends = []
            for op in tb.ops:
                op.depends = list(
                    filter(lambda dep: dep not in running_depends, op.depends))
                running_depends = running_depends + op.depends

    # Mark all ops that have a dependence on them
    has_dependence = set()
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            for op in tb.ops:
                has_dependence.update(op.depends)

    if dependence_nop:
        for gpu in program.gpus:
            for tb in gpu.threadblocks:
                pre_ops = []
                after_ops = []
                first_re = None
                first_dep = None
                for i, op in enumerate(tb.ops):
                    # Expand extra dependencies into nop operations
                    num_depends = len(op.depends)
                    if op.inst is Instruction.reduce:
                        if num_depends > 0:
                            for dep in op.depends:
                                if first_dep is None:
                                    first_dep = dep
                                else:
                                    pre_ops.append(Op(Instruction.nop, -1, None, None, [dep]))
                            op.depends = []
                        if first_re is None:
                            first_re = op

                    if first_re is not None:
                        after_ops.append(op)
                    else:
                        pre_ops.append(op)
                if first_dep is not None:
                    first_re.depends = [first_dep]
                tb.ops = pre_ops + after_ops

    # Do some additional postprocessing of operations:
    # - Expand operations with extra dependencies with no-ops
    # - Mark the index of each operation taking any extra no-ops into account
    op_idx = {}
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            new_ops = []
            for op in tb.ops:
                # Expand extra dependencies into nop operations
                if len(op.depends) > 1:
                    extra_deps = op.depends[1:]
                    op.depends = op.depends[:1]
                    for i, dep in enumerate(extra_deps):
                        new_ops.append(Op(Instruction.nop, -1, None, None, [dep]))
                        op_idx[new_ops[-1]] = len(new_ops) - 1
                        #op_tb_id[new_ops[-1]] = op_tb_id[op]
                new_ops.append(op)
                op_idx[new_ops[-1]] = len(new_ops) - 1
            tb.ops = new_ops

    nchannels = 0
    for gpu in program.gpus:
        max_tb_channels = 0
        if len(gpu.threadblocks) > 0:
            max_tb_channels = max(tb.channel+1 for tb in gpu.threadblocks)
        nchannels = max(nchannels, max_tb_channels)
    # Generate the XML structure
    algo_elem = ET.Element('algo')
    algo_elem.set('name', program.name)
    algo_elem.set('proto', program.protocol)
    algo_elem.set('nchannels', str(nchannels))
    if old_format:
        algo_elem.set('nchunksperloop', str(
            max(max(buffer_sizes[(gpu.rank, Buffer.input)], buffer_sizes[(gpu.rank, Buffer.output)]) for gpu in program.gpus)))
    algo_elem.set('ngpus', str(len(program.gpus)))
    algo_elem.set('coll', program.collective)
    algo_elem.set('inplace', str(1 if program.inplace else 0))
    for gpu in program.gpus:
        gpu_elem = ET.SubElement(algo_elem, 'gpu')
        gpu_elem.set('id', str(gpu.rank))
        gpu_elem.set('i_chunks', str(max(buffer_sizes[(gpu.rank, Buffer.input)], gpu.input_chunks)))
        gpu_elem.set('o_chunks', str(max(buffer_sizes[(gpu.rank, Buffer.output)], gpu.output_chunks)))
        gpu_elem.set('s_chunks', str(max(buffer_sizes[(gpu.rank, Buffer.scratch)], gpu.scratch_size())))
        for tb in gpu.threadblocks:
            tb_elem = ET.SubElement(gpu_elem, 'tb')
            tb_elem.set('id', str(tb_id[tb]))
            tb_elem.set('send', str(tb.send))
            tb_elem.set('recv', str(tb.recv))
            tb_elem.set('chan', str(tb.channel))
            for op in tb.ops:
                op_elem = ET.SubElement(
                    tb_elem, 'op' if not old_format else 'step')
                op_elem.set('step' if not old_format else 's', str(op_idx[op]))
                op_elem.set('type', str(op.inst))

                # The NCCL backend currently wants scratch at the end of output
                if not use_scratch:
                    if op.src.buffer == Buffer.scratch:
                        op.src.buffer = Buffer.output
                        op.src.index += buffer_sizes[(gpu.rank, Buffer.output)]
                    if op.dst_buffer == Buffer.scratch:
                        op.dst.buffer = Buffer.output
                        op.dst.index += buffer_sizes[(gpu.rank, Buffer.output)]

                if old_format:
                    if op.src is not None:
                        op_elem.set('srcbuf', str(op.src.buffer))
                        op_elem.set('srcoff', str(op.src.index))
                    else:
                        op_elem.set('srcbuf', 'i')
                        op_elem.set('srcoff', '-1')
                    if op.dst is not None:
                        op_elem.set('dstbuf', str(op.dst.buffer))
                        op_elem.set('dstoff', str(op.dst.index))
                    else:
                        op_elem.set('dstbuf', 'o')
                        op_elem.set('dstoff', '-1')
                else:
                    if op.is_send():
                        if op.src is not None:
                            op_elem.set('buf', str(op.src.buffer))
                            op_elem.set('off', str(op.src.index))
                    else:
                        if op.dst is not None:
                            op_elem.set('buf', str(op.dst.buffer))
                            op_elem.set('off', str(op.dst.index))
                if op.cnt() > 1 or old_format:
                    op_elem.set('cnt', str(op.cnt()))
                assert len(op.depends) <= 1
                if len(op.depends) == 1:
                    op_elem.set('depid', str(op_tb_id[op.depends[0]]))
                    op_elem.set('deps', str(op_idx[op.depends[0]]))
                elif old_format:
                    op_elem.set('depid', '-1')
                    op_elem.set('deps', '-1')
                if op in has_dependence:
                    op_elem.set('hasdep', '1')
                elif old_format:
                    op_elem.set('hasdep', '0')

    if pretty_print:
        ET.indent(algo_elem, space='  ')
    return ET.tostring(algo_elem, encoding='unicode')

def ir_to_json(program: Program, dependence_nop=False):
    # Figure out sizes of buffers based on usage
    buffer_sizes = defaultdict(lambda: 0)
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            for op in tb.ops:
                if op.inst in _local_src_insts_mscclpp:
                    key = (gpu.rank, op.src.buffer)
                    buffer_sizes[key] = max(
                        buffer_sizes[key], op.src.index + op.src.size)
                    for src in op.srcs:
                        key = (gpu.rank, src.buffer)
                        buffer_sizes[key] = max(
                            buffer_sizes[key], src.index + src.size)
                if op.inst in _local_dst_insts_mscclpp:
                    key = (gpu.rank, op.dst.buffer)
                    buffer_sizes[key] = max(
                        buffer_sizes[key], op.dst.index + op.dst.size)
                    # ignore remote buffers
                    if op.inst != Instruction.read_reduce_copy_send and op.inst != Instruction.reduce_send:
                        for dst in op.dsts:
                            key = (gpu.rank, dst.buffer)
                            buffer_sizes[key] = max(
                                buffer_sizes[key], dst.index + dst.size)
    for gpu in program.gpus:
        gpu.input_chunks = max(buffer_sizes[(gpu.rank, Buffer.input)], gpu.input_chunks)
        gpu.output_chunks = max(buffer_sizes[(gpu.rank, Buffer.output)], gpu.output_chunks)
        gpu.scratch_chunks = max(buffer_sizes[(gpu.rank, Buffer.scratch)], gpu.scratch_chunks)

    # get channel info for each GPU and threadblock
    for gpu in program.gpus:
        gpu.threadblocks = sorted(gpu.threadblocks, key=lambda tb: tb.id)
        chan_dict = {}
        # the channel key is the tuple (srcBuffer, dstBuffer, type)
        for tb in gpu.threadblocks:
            for ch in tb.channels:
                key = (ch.srcBuffer, ch.dstBuffer, ch.type)
                if key not in chan_dict:
                    chan_dict[key] = [(tb.id, ch.connected_to)]
                else:
                    chan_dict[key].append((tb.id, ch.connected_to))
        for key, value in chan_dict.items():
            chan_dict[key] = sorted(value)
        gpu.channels = chan_dict

    # Filter out dependencies within the same threadblock
    op_tb_id = {}
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            for op in tb.ops:
                op_tb_id[op] = op.tb
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            for op in tb.ops:
                op.depends = list(
                    filter(lambda dep: op_tb_id[dep] != op.tb, op.depends))
    # Filter out redundant dependencies
    # e.g. if op1 and op2 depend on op, and op1 happends before op2
    # then op2 does not need to explicitly depend on op
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            running_depends = []
            for op in tb.ops:
                op.depends = list(
                    filter(lambda dep: dep not in running_depends, op.depends))
                running_depends = running_depends + op.depends

    # Mark all ops that have a dependence on them
    has_dependence = set()
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            for op in tb.ops:
                has_dependence.update(op.depends)

    if dependence_nop:
        for gpu in program.gpus:
            for tb in gpu.threadblocks:
                pre_ops = []
                after_ops = []
                first_re = None
                first_dep = None
                for i, op in enumerate(tb.ops):
                    # Expand extra dependencies into nop operations
                    num_depends = len(op.depends)
                    if op.inst is Instruction.reduce:
                        if num_depends > 0:
                            for dep in op.depends:
                                if first_dep is None:
                                    first_dep = dep
                                else:
                                    pre_ops.append(Op(Instruction.nop, -1, None, None, [dep]))
                            op.depends = []
                        if first_re is None:
                            first_re = op

                    if first_re is not None:
                        after_ops.append(op)
                    else:
                        pre_ops.append(op)
                if first_dep is not None:
                    first_re.depends = [first_dep]
                tb.ops = pre_ops + after_ops

    # Do some additional postprocessing of operations:
    # - Expand operations with extra dependencies with no-ops
    # - Mark the index of each operation taking any extra no-ops into account
    op_idx = {}
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            new_ops = []
            for op in tb.ops:
                # Expand extra dependencies into nop operations
                if len(op.depends) > 1:
                    extra_deps = op.depends[1:]
                    op.depends = op.depends[:1]
                    for i, dep in enumerate(extra_deps):
                        new_ops.append(Op(Instruction.nop, -1, None, None, [dep]))
                        op_idx[new_ops[-1]] = len(new_ops) - 1
                        #op_tb_id[new_ops[-1]] = op_tb_id[op]
                new_ops.append(op)
                op_idx[new_ops[-1]] = len(new_ops) - 1
            tb.ops = new_ops

    # Need to calculate channel info for each GPU
    nchannels = 0
    for gpu in program.gpus:
        max_tb_channels = 0
        if len(gpu.threadblocks) > 0:
            max_tb_channels = max(tb.channel+1 for tb in gpu.threadblocks)
        nchannels = max(nchannels, max_tb_channels)
    return dump_to_json(program)

def dump_to_json(program: Program):
    gpus = []

    def get_channel_ids(chunk_list, tb_channel_dict, src_buffer, dst_buffer, chan_type):
        channel_ids = []
        for c in chunk_list:
            key = (src_buffer, dst_buffer, chan_type)
            channel_ids.extend([{"id": id, "off": c.index} for id, ele in enumerate(tb_channel_dict[key]["connectedTo"]) if ele == c.rank])
        return channel_ids

    for id, gpu in enumerate(program.gpus):
        gpu_instance = {
            'id': id,
            'input_chunks': gpu.input_chunks,
            'output_chunks': gpu.output_chunks,
            'scratch_chunks': gpu.scratch_chunks,
            'threadblocks': [],
            "channels": []
        }
        for (srcBuffer, dstBuffer, type), channels in gpu.channels.items():
            obj = {
                "srcBuffer": srcBuffer.name if hasattr(srcBuffer, 'name') else srcBuffer,
                "dstBuffer": dstBuffer.name if hasattr(dstBuffer, 'name') else dstBuffer,
                "type": type.name,
                "connectedTo": [eles[1] for eles in channels]
            }
            gpu_instance["channels"].append(obj)
        gpu_instance["channels"] = list(filter(lambda x: x["type"] != "none", gpu_instance["channels"]))
        for tb in gpu.threadblocks:
            if tb.id == -1:
                continue
            ops = []
            tb_channels = []
            tb_channel_dict = {}
            for (srcBuffer, dstBuffer, type), channels in gpu.channels.items():
                obj = {
                    "srcBuffer": srcBuffer.value if hasattr(srcBuffer, 'name') else srcBuffer,
                    "dstBuffer": dstBuffer.value if hasattr(dstBuffer, 'name') else dstBuffer,
                    "type": type.name,
                    "chanIds": [id for id, ele in enumerate(channels) if ele[0] == tb.id],
                    "connectedTo": [ele[1] for ele in channels if ele[0] == tb.id],
                }
                tb_channel_dict[(srcBuffer, dstBuffer, type)] = obj
                tb_channels.append(obj)
            tb_channels = filter(lambda x: x["type"] != "none", tb_channels)
            for op in tb.ops:
                if op.tb == -1:
                    continue
                if op.inst == Instruction.signal:
                    # get dst channel ids
                    dst_channel_ids = get_channel_ids(op.dsts, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type)
                    instr = {
                        "name": op.inst.name,
                        "o_cids": dst_channel_ids,
                        "srcbuff": op.src.buffer.value if op.src.buffer else None,
                        "dstbuff": op.dst.buffer.value if op.dst.buffer else None,
                        "ctype": op.channel_type.value,
                    }
                elif op.inst == Instruction.wait:
                    # get src channel ids
                    src_channel_ids = get_channel_ids(op.srcs, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type)
                    instr = {
                        "name": op.inst.name,
                        "i_cids": src_channel_ids,
                        "srcbuff": op.src.buffer.value if op.src.buffer else None,
                        "dstbuff": op.dst.buffer.value if op.dst.buffer else None,
                        "ctype": op.channel_type.value,
                    }
                elif op.inst == Instruction.read_reduce_copy:
                    src_channel_ids = get_channel_ids(op.srcs, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type)
                    instr = {
                        "name": op.inst.value,
                        "i_cids": src_channel_ids,
                        "srcbuff": op.src.buffer.value if op.src.buffer else None,
                        "dstbuff": op.dst.buffer.value if op.dst.buffer else None,
                        "dstoff": op.dst.index if op.dst else None,
                        "ctype": op.channel_type.value,
                        "cnt": op.cnt(),
                    }
                elif op.inst == Instruction.read_reduce_copy_send:
                    src_channel_ids = get_channel_ids(op.srcs, tb_channel_dict, op.src.buffer, op.dst.buffer, op.channel_type)
                    dst_channel_ids = get_channel_ids(op.dsts, tb_channel_dict, op.dst.buffer, op.dsts[0].buffer, op.channel_type)
                    instr = {
                        "name": op.inst.value,
                        "i_cids": src_channel_ids,
                        "o_cids": dst_channel_ids,
                        "srcbuff": op.src.buffer.value if op.src.buffer else None,
                        "dstbuff": op.dst.buffer.value if op.dst.buffer else None,
                        "dstoff": op.dst.index if op.dst else None,
                        "ctype": op.channel_type.value,
                        "cnt": op.cnt(),
                    }
                elif op.inst == Instruction.reduce_send:
                    dst_channel_ids = get_channel_ids(op.dsts, tb_channel_dict,  op.dst.buffer, op.dsts[0].buffer, ChannelType.sm)
                    instr = {
                        "name": op.inst.value,
                        "o_cids": dst_channel_ids,
                        "srcbuff": op.src.buffer.value if op.src.buffer else None,
                        "dstbuff": op.dst.buffer.value if op.dst.buffer else None,
                        "dstoff": op.dst.index if op.dst else None,
                        "srcs": list(map(lambda x: {"buff": x.buffer, "off": x.index}, op.srcs)),
                        "cnt": op.cnt(),
                    }
                elif op.inst == Instruction.reduce:
                    instr = {
                        "name": op.inst.value,
                        "srcbuff": op.src.buffer.value if op.src.buffer else None,
                        "dstbuff": op.dst.buffer.value if op.dst.buffer else None,
                        "dstoff": op.dst.index if op.dst else None,
                        "srcs": list(map(lambda x: {"buff": x.buffer, "off": x.index}, op.srcs)),
                        "cnt": op.cnt(),
                    }
                else:
                    instr = {
                        "name": op.inst.value,
                        "src": op.src.rank if op.src else None,
                        "srcbuff": op.src.buffer.value if op.src.buffer else None,
                        "srcoff": op.src.index if op.src else None,
                        "dst": op.dst.rank if op.dst else None,
                        "dstbuff": op.dst.buffer.value if op.dst.buffer else None,
                        "dstoff": op.dst.index if op.dst else None,
                        "ctype": op.channel_type.value,
                        "cnt": op.cnt(),
                    }
                ops.append(instr)
            threadblock = {
                'id': tb.id,
                'ops': ops,
                'channels': list(map(lambda x: {"src": x["srcBuffer"], "dst": x["dstBuffer"], "ctype": x["type"], "cid": x["chanIds"]}, tb_channels))
            }
            gpu_instance['threadblocks'].append(threadblock)
        gpus.append(gpu_instance)
    obj = {
        'name': program.name,
        'colletive': program.collective,
        'protocol': program.protocol,
        'inplace': program.inplace,
        'gpus': gpus
    }
    return json.dumps(obj, indent=2)
