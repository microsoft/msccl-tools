# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.collectives import *
from msccl.algorithm import *
from msccl.instance import *
from msccl.topologies import *

def _alltoall_subproblem(local_nodes, num_copies):
    remote_node = local_nodes

    local_end = local_nodes * local_nodes
    num_remote_pairs = (num_copies - 1) * local_nodes * local_nodes
    remote_out_end = local_end + num_remote_pairs
    num_chunks = remote_out_end + num_remote_pairs

    def cases(chunk, local,remote_out,remote_in):
        if chunk < local_end:
            return local(chunk)
        elif chunk < remote_out_end:
            return remote_out(chunk - local_end)
        else:
            return remote_in(chunk - remote_out_end)

    def pre(rank, chunk):
        return cases(chunk,
            lambda c: rank == c % local_nodes,
            lambda c: rank == (c // (num_copies - 1)) % local_nodes,
            lambda c: rank == remote_node)

    def post(rank, chunk):
        return cases(chunk,
            lambda c: rank == c // local_nodes,
            lambda c: rank == remote_node,
            lambda c: rank == (c // (num_copies - 1)) // local_nodes)

    def trigger(rank, chunk):
        if rank == remote_node:
            return cases(chunk,
                lambda c: None,
                lambda c: chunk + num_remote_pairs,
                lambda c: chunk - num_remote_pairs)
        else:
            return None

    return build_collective(f'AlltoallSubproblem(n={local_nodes},copies={num_copies})',
        local_nodes + 1, num_chunks,
        pre, post, trigger=trigger)

def make_alltoall_subproblem_collective_and_topology(topology, num_copies, relay_nodes, bw = 1, share_bw = False):
    local_nodes = topology.num_nodes()
    remote_node = local_nodes

    links = [[0 for _ in range(local_nodes + 1)] for _ in range(local_nodes + 1)]
    for src in range(local_nodes):
        for dst in range(local_nodes):
            links[dst][src] = topology.link(src, dst)
    for relay in relay_nodes:
        links[remote_node][relay] = bw
        links[relay][remote_node] = bw

    switches = topology.switches.copy()
    if share_bw:
        switches.append((relay_nodes, [num_nodes + 1], bw, 'remote_out'))
        switches.append(([num_nodes + 1], relay_nodes, bw, 'remote_in'))

    collective = _alltoall_subproblem(local_nodes, num_copies)
    topology = Topology(f'Subtopo(local={topology.name},relays=({",".join(str(i) for i in relay_nodes)}))', links, topology.switches)
    return collective, topology

def synthesize_alltoall_subproblem(subproblem_algo, num_copies, logging=False):
    if subproblem_algo.is_pipelined():
        raise ValueError('Pipelining is not supported.')

    local_topology = subproblem_algo.topology

    chunks = subproblem_algo.instance.chunks
    local_nodes = local_topology.num_nodes() - 1
    remote_node = local_nodes
    nodes = local_nodes * num_copies

    collective = alltoall(nodes).chunk_up(chunks)

    # Create a distributed topology where copies of relay nodes that connect to the remote node in the subproblem
    # topology are connected to all the relay nodes in the other copies.
    links = [[0 for _ in range(nodes)] for _ in range(nodes)]
    for dst in range(nodes):
        for src in range(nodes):
            local_src = src % local_nodes
            local_dst = dst % local_nodes
            if src // local_nodes != dst // local_nodes:
                bw = min(local_topology.link(local_src, remote_node), local_topology.link(remote_node, local_dst))
                links[dst][src] = bw
            else:
                links[dst][src] = local_topology.link(local_src, local_dst)

    # Also make copies of switches with a similar expansion of the remote node into the nodes of other copies.
    switches = []
    for srcs, dsts, bw, name in local_topology.switches:
        for i in range(num_copies):
            def to_dist(ranks):
                for rank in ranks:
                    if rank < remote_node:
                        # Non-remote nodes are just translated to the distributed numbering of ranks.
                        yield rank + i * local_nodes
                    else:
                        # Include all remote nodes in the switch. This is fine because the links already limit
                        # connectivity to just the relay nodes.
                        for r in range(nodes):
                            if r // local_nodes != i:
                                yield r

            dist_srcs = list(to_dist(srcs))
            dist_dsts = list(to_dist(dsts))
            switches.append((dist_srcs, dist_dsts, bw, f'copy_{i}_{name}_local'))

    topology = Topology(f'Stiched(sub={local_topology.name},copies={num_copies})', links, switches)

    def nth_chunk_for_pair(src, dst, idx):
        # The following chunk calculation respects both the _scattered and _transpose
        # pre/postconditions in Alltoall. When substituting it in:
        # -the precondition (chunk % self.num_nodes) simplifies to src
        # -the postcondition ((chunk // self.num_nodes) % self.num_nodes) simplifies to dst
        return (src + dst * collective.num_nodes) * chunks + idx

    steps = []

    # Calculate the ranges of the differently handled chunks
    local_end = local_nodes * local_nodes
    num_remote_pairs = (num_copies - 1) * local_nodes * local_nodes
    remote_out_end = local_end + num_remote_pairs
    num_chunks = remote_out_end + num_remote_pairs

    for local_step in subproblem_algo.steps:
        sends = []

        # These are used to track operations involving remote nodes that get matched with another operation in the same
        # step.
        unmatched_sends = {}
        unmatched_recvs = {}

        # Stitch together copies of the subproblem algorithm
        for chunk, src, dst in local_step.sends:
            for i in range(num_copies):
                def to_dist(rank):
                    # Translates ranks from the local to the distributed topology
                    return rank + i * local_nodes

                def other_start(c):
                    # Given a relative remote chunk return local rank 0 in the copy it corresponds to 
                    other_i = c % (num_copies - 1)
                    if other_i >= i:
                        other_i += 1
                    return other_i * local_nodes

                # Calculate origin and target ranks that match the Alltoall pre/postconditions
                if chunk < local_end:
                    assert src != remote_node and dst != remote_node

                    origin = to_dist((chunk // chunks) % local_nodes)
                    target = to_dist((chunk // chunks) // local_nodes)

                    # Check that the origin and target calculation match the local collective    
                    assert subproblem_algo.collective.precondition(origin % local_nodes, chunk)
                    assert subproblem_algo.collective.postcondition(target % local_nodes, chunk)
                elif chunk < remote_out_end:
                    c = chunk - local_end
                    local_origin = ((c // chunks) // (num_copies - 1)) % local_nodes

                    origin = to_dist(local_origin)
                    target = other_start(c) + ((c // (num_copies - 1))) // local_nodes

                    # Check that the origin and target calculation match the local collective
                    assert subproblem_algo.collective.precondition(local_origin, chunk)
                    assert subproblem_algo.collective.postcondition(target % local_nodes, chunk + num_remote_pairs)
                else:
                    assert chunk < num_chunks
                    c = chunk - remote_out_end
                    local_target = ((c // chunks) // (num_copies - 1)) // local_nodes
                    
                    target = to_dist(local_target)
                    origin = other_start(c) + ((c // (num_copies - 1))) % local_nodes

                    # Check that the origin and target calculation match the local collective
                    assert subproblem_algo.collective.precondition(origin % local_nodes, chunk - num_remote_pairs)
                    assert subproblem_algo.collective.postcondition(local_target, chunk)
                
                # Get the chunk number in the distributed algorithm
                chunk_idx = chunk % chunks
                # Translate send src and dst to distributed space and add the send to the distributed algorithm
                dist_chunk = nth_chunk_for_pair(origin, target, chunk_idx)

                if dst == remote_node:
                    assert chunk < remote_out_end
                    # Sends to remote nodes have to find a matched receive
                    if dist_chunk in unmatched_recvs:
                        dist_dst = unmatched_recvs.pop(dist_chunk)
                        sends.append((dist_chunk, to_dist(src), dist_dst))
                    else:
                        unmatched_sends[dist_chunk] = to_dist(src)
                elif src == remote_node:
                    assert chunk < num_chunks
                    # Receives from remote nodes have to find a matched send
                    if dist_chunk in unmatched_sends:
                        dist_src = unmatched_sends.pop(dist_chunk)
                        sends.append((dist_chunk, dist_src, to_dist(dst)))
                    else:
                        unmatched_recvs[dist_chunk] = to_dist(dst)
                else:
                    # Sends locally are just translated to the new distributed space of ranks
                    sends.append((dist_chunk, to_dist(src), to_dist(dst)))

        if len(unmatched_sends) > 0 or len(unmatched_recvs) > 0:
            raise ValueError('Subproblem algorithm has unpaired sends/recvs.')

        steps.append(Step(local_step.rounds, sends))

    instance = Instance(
        steps=len(steps),
        extra_rounds=sum(step.rounds - 1 for step in steps),
        chunks=chunks,
    )
    return Algorithm.make_implementation(collective, topology, instance, steps)
