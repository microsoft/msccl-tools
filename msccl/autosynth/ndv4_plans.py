# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.autosynth.registry import register_synthesis_plan, register_msccl_program
from msccl.programs.allreduce_a100_ring import allreduce_ring
from msccl.programs.allreduce_allpairs import allreduce_allpairs
from msccl.programs.alltoall_a100_yifan import alltoall_hierarchical
from msccl.programs.alltoall_a100_8kp1 import alltoall_three_step
from msccl.topologies import fully_connected
from msccl.language.ir import ThreadblockPolicy

def register_ndv4_plans():

    @register_msccl_program(fully_connected(8), 'allreduce', 'ndv4', chunk_factor=64, inplace=True,
        instances=2, protocol='LL', sizes=('512B', '82944B'), threadblock_policy=ThreadblockPolicy.manual, interleaved_replication=False, dependence_nop=True, machines= lambda x: x == 1)
    def ndv4_allpairs_allreduce_config1(prog, nodes):
        allreduce_allpairs(8)

    @register_msccl_program(fully_connected(8), 'allreduce', 'ndv4', chunk_factor=64, inplace=True,
        instances=4, protocol='LL', sizes=('82944B', '458752B'), threadblock_policy=ThreadblockPolicy.manual, interleaved_replication=False, dependence_nop=True, machines= lambda x: x == 1)
    def ndv4_allpairs_allreduce_config2(prog, nodes):
        allreduce_allpairs(8)

    @register_msccl_program(fully_connected(8), 'allreduce', 'ndv4', chunk_factor=8, inplace=True,
        instances=8, protocol='LL', sizes=('458752B', '2129920B'), threadblock_policy=ThreadblockPolicy.manual, machines= lambda x: x == 1)
    def ndv4_ring_allreduce_config1(prog, nodes):
        allreduce_ring(size=8, channels=4)

    @register_msccl_program(fully_connected(8), 'allreduce', 'ndv4', chunk_factor=8, inplace=True,
        instances=8, protocol='LL128', sizes=('2129920B', '22806528B'), threadblock_policy=ThreadblockPolicy.manual, machines= lambda x: x == 1)
    def ndv4_ring_allreduce_config2(prog, nodes):
        allreduce_ring(size=8, channels=4)

    @register_msccl_program(fully_connected(8), 'alltoall', 'ndv4', protocol='LL128', sizes=('1MB', '32MB'), machines=lambda x: x == 8 or x == 16 or x == 32 or x == 64)
    def ndv4_alltoall_hierarchical_config1(prog, nodes):
        alltoall_hierarchical(num_nodes=nodes, gpus_per_node=8)

    @register_msccl_program(fully_connected(8), 'alltoall', 'ndv4', protocol='Simple', sizes=('32MB', None), machines=lambda x: x == 8 or x == 16 or x == 32)
    def ndv4_alltoall_hierarchical_config2(prog, nodes):
        alltoall_hierarchical(num_nodes=nodes, gpus_per_node=8)

    @register_msccl_program(fully_connected(8), 'alltoall', 'ndv4', protocol='Simple', sizes=('32MB', None), machines=lambda x: x == 64)
    def ndv4_alltoall_three_step(prog, nodes):
        alltoall_three_step(num_nodes=nodes, gpus_per_node=8)

    @register_msccl_program(fully_connected(8), 'alltoall', 'ndv4', protocol='Simple', sizes=('1KB', None), machines=lambda x: x == 2 or x == 4)
    def ndv4_alltoall_hierarchical_config2(prog, nodes):
        alltoall_hierarchical(num_nodes=nodes, gpus_per_node=8)

        
