# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import msccl.topologies as topologies
import pathlib

class KnownDistributedTopologies:
    def __init__(self, parser):
        self.parser = parser
        self.constructors = {
            'DistributedFullyConnected': topologies.distributed_fully_connected,
            'DistributedHubAndSpoke': topologies.distributed_hub_and_spoke,
        }
        self.parser.add_argument('topology', type=str, choices=self.constructors.keys(), help='the distributed topology')
        self.parser.add_argument('-n', '--nodes', type=int, help='total nodes in the distributed topology, must be divisible by local topology')
        self.parser.add_argument('--copies', type=int, help='copies of the local topology to be made')
        self.parser.add_argument('-bw', '--remote-bandwidth', type=int, default=1, help='bandwidth of links in the distributed topology', metavar='N')

    def create(self, args, local_topology):
        if args.nodes != None and args.copies != None:
            self.parser.error('please use only one of -n/--nodes, --copies')
        if args.copies != None:
            copies = args.copies
        elif args.nodes != None:
            if args.nodes % local_topology.num_nodes() != 0:
                self.parser.error(f'total number of nodes must be divisible by the local number of nodes {local_topology.num_nodes()}, but {args.nodes} was given')
            copies = args.nodes // local_topology.num_nodes()
        else:
            self.parser.error('one of the following arguments is required: --nodes, --copies')
        return self.constructors[args.topology](local_topology, copies, args.remote_bandwidth)
