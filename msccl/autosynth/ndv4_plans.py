# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.autosynth.registry import register_synthesis_plan, register_msccl_program
from msccl.programs.allreduce_a100_ring import allreduce_ring
from msccl.programs.alltoall_a100_yifan import alltoall_hierarchical
from msccl.programs.alltoall_a100_8kp1 import alltoall_three_step
from msccl.topologies import fully_connected
from msccl.language.ir import ThreadblockPolicy

def register_ndv4_plans():

    @register_msccl_program(fully_connected(8), 'allreduce', 'ndv4', chunk_factor=8, inplace=True,
        instances=4, protocol='LL128', sizes=('256KB', '20MB'), threadblock_policy=ThreadblockPolicy.manual, machines= lambda x: x == 1)
    def ndv4_ring_allreduce(prog, nodes):
        allreduce_ring(size=8, channels=8)

    @register_msccl_program(fully_connected(8), 'alltoall', 'ndv4', protocol='LL128', sizes=('1MB', '32MB'), machines=lambda x: x == 8 or x == 16 or x == 32 or x == 64)
    def ndv4_alltoall_hierarchical(prog, nodes):
        alltoall_hierarchical(num_nodes=nodes, gpus_per_node=8)

    @register_msccl_program(fully_connected(8), 'alltoall', 'ndv4', protocol='Simple', sizes=('32MB', None), machines=lambda x: x == 8 or x == 16 or x == 32)
    def ndv4_alltoall_hierarchical(prog, nodes):
        alltoall_hierarchical(num_nodes=nodes, gpus_per_node=8)

    @register_msccl_program(fully_connected(8), 'alltoall', 'ndv4', protocol='Simple', sizes=('32MB', None), machines=lambda x: x == 64)
    def ndv4_alltoall_three_step(prog, nodes):
        alltoall_three_step(num_nodes=nodes, gpus_per_node=8)
