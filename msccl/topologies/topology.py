# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Tuple

Switch = Tuple[List[int], List[int], int, str]

class Topology(object):
    def __init__(self, name: str, links: List[List[int]], switches: List[Switch]=[]):
        self.name = name
        self.links = links
        self.switches: List = switches
        for srcs, dsts, bw, switch_name in switches:
            if bw == 0:
                raise ValueError(f'Switch {switch_name} has zero bandwidth, but switch bandwidths must be strictly positive. Please encode connectedness in links.')
            if bw < 0:
                raise ValueError(f'Switch {switch_name} has a negative bandwidth of {bw}. Bandwidth must be strictly positive.')

    def sources(self, dst: int):
        for src, bw in enumerate(self.links[dst]):
            if bw > 0:
                yield src

    def destinations(self, src: int):
        for dst, links in enumerate(self.links):
            bw = links[src]
            if bw > 0:
                yield dst

    def link(self, src: int, dst: int):
        return self.links[dst][src]

    def num_nodes(self):
        return len(self.links)

    def nodes(self):
        return range(self.num_nodes())
    
    def bandwidth_constraints(self):
        for dst, dst_links in enumerate(self.links):
            for src, bw in enumerate(dst_links):
                if bw > 0:
                    yield ([src], [dst], bw, f'{src}→{dst}')
        for srcs, dsts, bw, switch_name in self.switches:
            yield (srcs, dsts, bw, switch_name)
