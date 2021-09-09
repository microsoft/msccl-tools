# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .topology import Topology

def _copy_links(remote_bw, num_local, num_dist, local_links):
    return [[remote_bw if src // num_local != dst // num_local else local_links[dst % num_local][src % num_local]
        for src in range(num_dist)] for dst in range(num_dist)]

def _copy_switches(num_local, num_copies, local_switches):
    switches = []
    for srcs, dsts, bw, name in local_switches:
        for i in range(num_copies):
            dist_srcs = [src + i * num_local for src in srcs]
            dist_dsts = [dst + i * num_local for dst in dsts]
            switches.append((dist_srcs, dist_dsts, bw, f'copy_{i}_{name}_local'))
    return switches

def _copy_links_ext(remote_lk_func, num_local, num_dist, local_links):
    return [[remote_lk_func(src,dst) if src // num_local != dst // num_local else local_links[dst % num_local][src % num_local]
        for src in range(num_dist)] for dst in range(num_dist)]

def _copy_invbw_ext(remote_invbw_func, num_local, num_dist, local_invbws):
    return [[remote_invbw_func(src,dst) if src // num_local != dst // num_local else local_invbws[dst % num_local][src % num_local]
        for src in range(num_dist)] for dst in range(num_dist)]

# Add switches over IB
def _add_ext_switches(num_local, senders, receivers, remote_invbw):
    switches = []
    for sender in senders:
        others_recv = [other for other in receivers if other//num_local != sender//num_local]
        switches.append(([sender],others_recv,1,f'node_{sender}_out'))
    for receiver in receivers:
        others_send = [other for other in senders if other//num_local != receiver//num_local]
        switches.append((others_send,[receiver],1,f'node_{receiver}_in'))
    return switches

def distributed_fully_connected(local_topology, num_copies, remote_bw):
    num_local = local_topology.num_nodes()
    num_dist = num_local * num_copies

    links = _copy_links(remote_bw, num_local, num_dist, local_topology.links)
    switches = _copy_switches(num_local, num_copies, local_topology.switches)

    return Topology(f'DistributedFullyConnected(local={local_topology.name},copies={num_copies},bw={remote_bw})', links, switches)

def distributed_hub_and_spoke(local_topology, num_copies, remote_bw):
    num_local = local_topology.num_nodes()
    num_dist = num_local * num_copies

    links = _copy_links(remote_bw, num_local, num_dist, local_topology.links)
    switches = _copy_switches(num_local, num_copies, local_topology.switches)

    for i in range(num_copies):
        local_ranks = [j + i * num_local for j in range(num_local)]
        remote_ranks = [k for k in range(num_dist) if k // num_local != i]
        switches.append((local_ranks, remote_ranks, remote_bw, f'copy_{i}_out_remote'))
        switches.append((remote_ranks, local_ranks, remote_bw, f'copy_{i}_in_remote'))
    
    return Topology(f'DistributedHubAndSpoke(local={local_topology.name},copies={num_copies},bw={remote_bw})', links, switches)

# Connects all external relay GPUs to each other with switches
# This prevents a GPU from receiving data from multiple GPUs over IB at the same time
# Supports only topologies which have defined alpha and beta costs
def dgx2_cluster(num_copies):
    relays = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
    def hub_and_spoke_nvswitch(num_nodes, num_switches, invbw, remote_invbw, name):
        links = [[0 if x==y else num_switches for y in range(num_nodes)] for x in range(num_nodes)]
        invbws = [[0 if x==y else invbw for y in range(num_nodes)] for x in range(num_nodes)]
        switches = []
        for i in range(num_switches):
            swt = []
            for node in range(num_nodes):
                others = [other for other in range(num_nodes) if other != node]
                swt.append(([node],others,1,f'node_{node}_out'))
                swt.append((others,[node],1,f'node_{node}_in'))
            switches.extend(swt)
        topo = Topology(f'HubAndSpoke{name}(n={num_nodes})', links, switches)
        topo.invbws = invbws
        topo.remote_invbw = remote_invbw
        return topo

    local_topology = hub_and_spoke_nvswitch(16,6,46,107,"DGX2")

    num_local = local_topology.num_nodes()
    num_dist = num_local * num_copies
    senders = [i*num_local + snd for i in range(num_copies) for snd in relays[0]]
    receivers = [i*num_local + rcv for i in range(num_copies) for rcv in relays[1]]
    links = _copy_links_ext(lambda src,dst: 1 if dst % num_local in relays[1] and src % num_local in relays[0] else 0,
        num_local, num_dist, local_topology.links)
    invbws = _copy_invbw_ext(lambda src,dst: local_topology.remote_invbw if dst % num_local in relays[1] and src % num_local in relays[0] else 0,
        num_local, num_dist, local_topology.invbws)
    switches = _copy_switches(num_local, num_copies, local_topology.switches)
    ext_switches = _add_ext_switches(num_local, senders, receivers, local_topology.remote_invbw)
    switches.extend(ext_switches)
    result = Topology(f'DistributedRelayedSwitch(local={local_topology.name},copies={num_copies})', links, switches)
    result.copies = num_copies
    return result