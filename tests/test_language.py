# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sccl
from sccl.topologies import line, fully_connected
from sccl.language import *
from sccl.language.collectives import *
import os
import pytest

class Send(Collective):
    # Initial state is chunk0 is on rank0 in the input buffer
    def init_buffers(self):
        chunks_per_node = self.chunk_factor
        rank_buffers = []
        for r in range(self.num_ranks):
            input_buffer = [None] * chunks_per_node
            output_buffer = [None] * chunks_per_node 
            if r == 0:
                for c in range(chunks_per_node):
                    input_buffer[c] = Chunk(r, c, 2, c)
            buffers = {Buffer.input : input_buffer, 
                    Buffer.output : output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers
            

    # Final state chunk0 from rank0 is in the output buffer of rank2
    def check(self, prog):
        correct = True
        output = prog.buffers[2][Buffer.output]
        for c in range(self.chunk_factor):
            chunk = output[c]
            # Check that we got chunk 0 from rank 0
            if chunk is None or chunk.origin_rank != 0 or chunk.origin_index != 0:
                print(f'Rank 2 chunk {c} is incorrect should be ({0}, {0}) given {chunk}')
                correct = False
        return correct

class Reduce(Collective):
    # Initial state is chunk0,0 is on rank0 in the input buffer
    # and chunk0,1 is on rank1 in the input buffer, etc.
    def init_buffers(self):
        chunks_per_node = self.chunk_factor
        rank_buffers = []
        for r in range(self.num_ranks):
            input_buffer = [None] * chunks_per_node
            output_buffer = [None] * chunks_per_node 
            for c in range(chunks_per_node):
                input_buffer[c] = Chunk(r, c, -1, c)
            buffers = {Buffer.input : input_buffer, 
                    Buffer.output : output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers
            

    # Final state rank2 has a fully reduced chunk from gpus 0, 1, and 2
    def check(self, prog):
        expected_chunk = ReduceChunk([])
        for r in range(self.num_ranks):
            expected_chunk = expected_chunk.reduce(Chunk(r, 0))

        correct = True
        chunk = prog.buffers[2][Buffer.input][0]
        if chunk is None or chunk != expected_chunk:
            print(f'Rank 2 chunk 0 is incorrect should be ReduceChunk index 0 from all ranks, given {chunk}')
            correct = False
        return correct

def test_send():
    num_gpus = 3
    topology = line(num_gpus)

    chunksperloop = 1
    instances = 1
    collective = Send(num_gpus, chunksperloop, inplace=False)
    with SCCLProgram("send", topology, collective, instances):
        chunk(0, Buffer.input, 0).send(1, 'scratch').send(2, Buffer.output, 0)
        assert Check()

def test_reduce():
    num_gpus = 3
    topology = line(num_gpus)

    chunksperloop = 1
    instances = 1
    collective = Reduce(num_gpus, chunksperloop, inplace=True)
    with SCCLProgram("reduce", topology, collective, instances):
        chunk(0, Buffer.input, 0).reduce(1, Buffer.input, 0).reduce(2, Buffer.input, 0)
        assert Check()

def test_local_copy():
    num_gpus = 3
    topology = fully_connected(num_gpus)

    chunksperloop = 1
    instances = 1
    collective = Send(num_gpus, chunksperloop, inplace=False)
    with SCCLProgram("cpy", topology, collective, instances):
        chunk(0, Buffer.input, 0).send(2, 'scratch').send(2, Buffer.output, 0)
        assert Check()

def test_local_reduce():
    num_gpus = 3
    topology = line(num_gpus)

    chunksperloop = 1
    instances = 1
    collective = Reduce(num_gpus, chunksperloop, inplace=True)
    with SCCLProgram("local-reduce", topology, collective, instances):
        chunk(0, Buffer.input, 0).reduce(1, Buffer.input, 0).send(2, 'scratch', 0).reduce(2, Buffer.input, 0)

        XML()
        assert Check()

def test_scratch_buffers():
    num_gpus = 3
    topology = fully_connected(num_gpus)

    chunksperloop = num_gpus
    instances = 1
    collective = AllReduce(num_gpus, chunksperloop, inplace=False)
    with SCCLProgram("test", topology, collective, instances):
        chunk(0, Buffer.input, 0).send(2, 'scratch', 2)
        c = chunk(2, 'scratch', 2)
        assert c.index == 2
        c = chunk(1, Buffer.input, 0).send(2, 'scratch')
        assert c.index == 3
        XML()

def test_allgather():
    topology = fully_connected(2)
    collective = AllGather(2, 1, True)
    with SCCLProgram("allgather", topology, collective, 1):
        chunk(0, Buffer.input, 0).send(1, Buffer.output, 0)
        chunk(1, Buffer.input, 0).send(0, Buffer.output, 1)
        assert Check()

def test_reducescatter():
    topology = fully_connected(2)
    collective = ReduceScatter(2, 1, True)
    with SCCLProgram("reducescatter", topology, collective, 1):
        chunk(0, Buffer.input, 1).reduce(1, Buffer.input, 1)
        chunk(1, Buffer.input, 0).reduce(0, Buffer.input, 0)
        assert Check()


def test_alltoall():
    topology = fully_connected(2)
    collective = AllToAll(2, 1, False)
    with SCCLProgram("alltoall", topology, collective, 1):
        chunk(0, Buffer.input, 0).send(0, Buffer.output, 0)
        chunk(0, Buffer.input, 1).send(1, Buffer.output, 0)
        chunk(1, Buffer.input, 0).send(0, Buffer.output, 1)
        chunk(1, Buffer.input, 1).send(1, Buffer.output, 1)
        assert Check()

def test_allreduce():
    topology = fully_connected(2)
    collective = AllReduce(2, 2, True)
    with SCCLProgram("allreduce", topology, collective, 1):
        chunk(0, Buffer.input, 0).reduce(1, Buffer.output, 0).send(0, Buffer.input, 0)
        chunk(1, Buffer.input, 1).reduce(0, Buffer.input, 1).send(1, Buffer.input, 1)
        assert Check()

def test_instruction_fusion():
    topology = fully_connected(3)
    collective = AllReduce(3, 3, True)
    prgm = SCCLProgram("allreduce", topology, collective, 1, threadblock_policy=ThreadblockPolicy.manual)
    with prgm:
        c = chunk(0, Buffer.input, 0, 3).reduce(1, Buffer.input, 0,sendtb=0, recvtb=0).reduce(2, Buffer.input, 0, sendtb=0, recvtb=0)
        c.send(0, Buffer.input, 0, sendtb=0, recvtb=0).send(1, Buffer.input, 0, sendtb=0, recvtb=0)
        assert Check()
    lowered_prgm = prgm.lower()
    assert lowered_prgm.gpus[0].threadblocks[0].ops[0].inst == Instruction.send
    assert lowered_prgm.gpus[0].threadblocks[0].ops[1].inst == Instruction.recv_copy_send
    assert lowered_prgm.gpus[1].threadblocks[0].ops[0].inst == Instruction.recv_reduce_send
    assert lowered_prgm.gpus[1].threadblocks[0].ops[1].inst == Instruction.recv
    assert lowered_prgm.gpus[2].threadblocks[0].ops[0].inst == Instruction.recv_reduce_copy_send

def test_replication():
    topology = fully_connected(2)
    collective = AllToAll(2, 1, False)
    prgm = SCCLProgram("alltoall", topology, collective, 1)
    with prgm:
        chunk(0, Buffer.input, 0).send(0, Buffer.output, 0)
        chunk(0, Buffer.input, 1).send(1, Buffer.output, 0)
        chunk(1, Buffer.input, 0).send(0, Buffer.output, 1)
        chunk(1, Buffer.input, 1).send(1, Buffer.output, 1)

    instances = 2
    replicated_prgm = SCCLProgram("alltoall", topology, collective, instances)
    with replicated_prgm:
            chunk(0, Buffer.input, 0).send(0, Buffer.output, 0)
            chunk(0, Buffer.input, 1).send(1, Buffer.output, 0)
            chunk(1, Buffer.input, 0).send(0, Buffer.output, 1)
            chunk(1, Buffer.input, 1).send(1, Buffer.output, 1)

    lowered_prgm = prgm.lower()
    lowered_replicated_prgm = replicated_prgm.lower()

    for gpu1, gpu2 in zip(lowered_prgm.gpus, lowered_replicated_prgm.gpus):
        assert len(gpu1.threadblocks) * instances == len(gpu2.threadblocks)

def test_illegal_tb_assignment():
    num_gpus = 3
    topology = fully_connected(num_gpus)
    collective = AllToAll(num_gpus, 1, False)
    prgm = SCCLProgram("alltoall", topology, collective, 1, threadblock_policy=ThreadblockPolicy.manual)
    with prgm:
        with pytest.raises(Exception):
            # Cannot send to two different gpus on the same threadblock
            chunk(0, Buffer.input, 0).send(1, Buffer.output, 0, sendtb=0, recvtb=1)
            chunk(0, Buffer.input, 1).send(2, Buffer.output, 0, sendtb=0, recvtb=1)
            XML()

def test_registered_alltoall_yifan():
    from sccl.programs.alltoall_a100_yifan import alltoall_hierarchical 

    num_nodes = 4
    gpus_per_node = 8
    num_ranks = num_nodes * gpus_per_node
    topology = fully_connected(num_ranks)
    collective = AllToAll(num_ranks, 1, inplace=False)
    with SCCLProgram("hierarchical_all_to_all", topology, collective, 1):
        alltoall_hierarchical(num_nodes, gpus_per_node)
        assert Check()

def test_registered_alltoall_8kp1():
    from sccl.programs.alltoall_a100_8kp1 import alltoall_hierarchical 

    num_nodes = 9
    gpus_per_node = 8
    num_ranks = num_nodes * gpus_per_node
    topology = fully_connected(num_ranks)
    collective = AllToAll(num_ranks, 1, inplace=False)
    with SCCLProgram("hierarchical_all_to_all", topology, collective, 1):
        alltoall_hierarchical(num_nodes, gpus_per_node)
        assert Check()
        XML()

def test_registered_allreduce():
    from sccl.programs.allreduce_a100_ring import allreduce_ring 

    num_ranks = 8
    instances = 4
    topology = fully_connected(num_ranks)
    collective = AllReduce(num_ranks, num_ranks, inplace=True)
    with SCCLProgram(f"allreduce", topology, collective, instances,
        protocol="LL128", threadblock_policy=ThreadblockPolicy.manual):
        allreduce_ring(num_ranks, num_ranks)
        assert Check()
        XML()
