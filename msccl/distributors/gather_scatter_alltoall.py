# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.collectives import *
from msccl.algorithm import *
from msccl.instance import *
from msccl.topologies import distributed_fully_connected

def synthesize_gather_scatter_distributed_alltoall(num_copies, gather_algo, scatter_algo, remote_bw=1, logging=False):
    if gather_algo.is_pipelined() or scatter_algo.is_pipelined():
        raise ValueError('Pipelining is not supported.')

    if gather_algo.instance.chunks != scatter_algo.instance.chunks:
        raise ValueError(f'Local gather and local scatter must have the same chunks (got {gather_algo.instance.chunks} and {scatter_algo.instance.chunks})')

    if gather_algo.topology.name != scatter_algo.topology.name:
        # TODO: improve this check to check actual structure, not just name
        raise ValueError(f'Local gather and local scatter must have the same topology (got {gather_algo.topology.name} and {scatter_algo.topology.name})')
    local_topology = gather_algo.topology

    chunks = gather_algo.instance.chunks
    local_nodes = gather_algo.topology.num_nodes()
    nodes = local_nodes * num_copies

    # Figure out the roots of the (possibly multi-root) gather
    gather_roots = []
    for chunk in range(gather_algo.collective.num_chunks // local_nodes):
        for rank in range(local_nodes):
            if gather_algo.collective.postcondition(rank, chunk):
                gather_roots.append(rank)
                break
        else:
            raise ValueError(f'Root number {chunk} not found for the Gather algorithm.')

    # Check that we got the roots right
    if len(gather_roots) > 1:
        local_gather = multiroot_gather(local_nodes, roots=gather_roots).chunk_up(chunks)
        try:
            gather_algo.check_implements(local_gather)
        except:
            raise ValueError(f'Given local Gather algorithm "{gather_algo.name}" does not implement MultirootGather for the roots {gather_roots}.')
    elif len(gather_roots) == 1:
        local_gather = gather(local_nodes, root=gather_roots[0]).chunk_up(chunks)
        try:
            gather_algo.check_implements(local_gather)
        except:
            raise ValueError(f'Given local Gather algorithm "{gather_algo.name}" does not implement Gather for the root {gather_roots[0]}.')
    else:
        raise ValueError(f'No roots found for the Gather algorithm.')

    # Figure out the roots of the (possibly multi-root) gather
    scatter_roots = []
    for chunk in range(scatter_algo.collective.num_chunks // local_nodes):
        for rank in range(local_nodes):
            if scatter_algo.collective.precondition(rank, chunk):
                scatter_roots.append(rank)
                break
        else:
            raise ValueError(f'Root number {chunk} not found for the Scatter algorithm.')

    # Check that we got the roots right
    if len(scatter_roots) > 1:
        local_scatter = multiroot_scatter(local_nodes, roots=scatter_roots).chunk_up(chunks)
        try:
            scatter_algo.check_implements(local_scatter)
        except:
            raise ValueError(f'Given local Scatter algorithm "{scatter_algo.name}" does not implement MultirootScatter for the roots {scatter_roots}.')
    elif len(scatter_roots) == 1:
        local_scatter = scatter(local_nodes, root=scatter_roots[0]).chunk_up(chunks)
        try:
            scatter_algo.check_implements(local_scatter)
        except:
            raise ValueError(f'Given local Scatter algorithm "{scatter_algo.name}" does not implement Scatter for the root {scatter_roots[0]}.')
    else:
        raise ValueError(f'No roots found for the Scatter algorithm.')

    if len(gather_roots) != len(scatter_roots):
        raise ValueError(f'The number of roots for the Gather algorithm ({len(gather_roots)}) does not match the number of roots for the Scatter algorithm ({len(scatter_roots)}).')

    # Multiply chunks to match the number of roots
    if len(gather_roots) > 1:
        chunks *= len(gather_roots)
        print(f'Multiplying chunks by {len(gather_roots)} to match the number of roots.')

    collective = alltoall(nodes)
    topology = distributed_fully_connected(gather_algo.topology, num_copies, remote_bw)

    def nth_chunk_for_pair(src, dst, idx):
        # The following chunk calculation respects both the _scattered and _transpose
        # pre/postconditions in Alltoall. When substituting it in:
        # -the precondition (chunk % self.num_nodes) simplifies to src
        # -the postcondition ((chunk // self.num_nodes) % self.num_nodes) simplifies to dst
        return (src + dst * nodes) * chunks + idx

    steps = []

    chunk_end = defaultdict(lambda: 0)

    for step_idx, local_step in enumerate(gather_algo.steps):
        sends = []

        # Translate copies of the local Gather to the new space of ranks
        for chunk, src, dst in local_step.sends:
            for target_rank in range(nodes):
                for i in range(num_copies):
                    # Translates ranks from the local to the distributed topology
                    def to_dist(rank):
                        return rank + i * local_nodes

                    # Calculate origin rank that matches the Gather precondition
                    origin = (chunk // chunks) % local_nodes

                    # Check that we got that calculation right
                    assert local_gather.precondition(origin, chunk)

                    # Get the chunk number in the distributed algorithm
                    chunk_idx = chunk % chunks
                    dist_chunk = nth_chunk_for_pair(to_dist(origin), target_rank, chunk_idx)

                    # Translate send src and dst to distributed space and the send to the distributed algorithm
                    sends.append((dist_chunk, to_dist(src), to_dist(dst)))
                    assert to_dist(src) != to_dist(dst)

                    # Update the latest step this chunk was touched on
                    chunk_end[dist_chunk] = max(chunk_end[dist_chunk], step_idx+1)

        steps.append(Step(local_step.rounds * nodes, sends))

    # Perform transpose between local root nodes
    transpose_sends = [[] for _ in range(len(gather_algo.steps) + 1)]
    for src in range(nodes):
        for dst in range(nodes):
                # Sends are needed for the chunks going from src to dst if they are in different copies or if the
                # gather and scatter roots are different.
                for chunk_idx in range(chunks):
                    gather_root = gather_roots[chunk_idx % len(gather_roots)]
                    scatter_root = scatter_roots[chunk_idx % len(scatter_roots)]
                    if (src // local_nodes == dst // local_nodes and
                            gather_root != scatter_root and
                            local_topology.link(gather_root, scatter_root) == 0):
                        raise ValueError(f'The local topology does not have a link from root {gather_root} of the Gather to root {scatter_root} of the Scatter.')
                    if gather_root != scatter_root or src // local_nodes != dst // local_nodes:
                        chunk = nth_chunk_for_pair(src, dst, chunk_idx)
                        # Calculate the local root ranks' global indices
                        root_src = (src // local_nodes) * local_nodes + gather_root
                        root_dst = (dst // local_nodes) * local_nodes + scatter_root
                        transpose_sends[chunk_end[chunk]].append((chunk, root_src, root_dst))
                        assert root_src != root_dst
    for i, sends in enumerate(transpose_sends):
        if i < len(gather_algo.steps):
            steps[i].sends.extend(sends)
            steps[i].rounds = max(steps[i].rounds, chunks * local_nodes * local_nodes)
        else:
            steps.append(Step(chunks * local_nodes * local_nodes, sends))

    #TODO: integrate into above
    if gather_root != scatter_root and local_topology.link(gather_root, scatter_root) == 0:
        raise ValueError(f'Local topology does not have a link from the root of the Gather ({gather_root}) to that of the Scatter ({scatter_root}).')

    for local_step in scatter_algo.steps:
        sends = []

        # Translate copies of the local Scatter to the new space of ranks
        for chunk, src, dst in local_step.sends:
            for source_rank in range(nodes):
                for i in range(num_copies):
                    # Translates ranks from the local to the distributed topology
                    def to_dist(rank):
                        return rank + i * local_nodes
                    
                    # Calculate target rank that matches the Scatter postcondition
                    target = (chunk // chunks) % local_nodes

                    # Check that we got that calculation right
                    assert local_scatter.postcondition(target, chunk)

                    # Get the chunk number in the distributed algorithm
                    chunk_idx = chunk % chunks
                    dist_chunk = nth_chunk_for_pair(source_rank, to_dist(target), chunk_idx)

                    # Translate send src and dst to distributed space and the send to the distributed algorithm
                    sends.append((dist_chunk, to_dist(src), to_dist(dst)))

        steps.append(Step(local_step.rounds * nodes, sends))

    instance = Instance(
        steps=len(steps),
        extra_rounds=sum(step.rounds - 1 for step in steps),
        chunks=chunks,
    )
    return Algorithm.make_implementation(collective, topology, instance, steps)
