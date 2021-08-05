# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.language import *
from sccl.topologies import *

num_nodes = 9
gpus_per_node = 8

def NodeGpuPairFromRank(r):
    return (r//gpus_per_node, r%gpus_per_node)

def RankFromNodeGpuPair(n, g):
    return n*gpus_per_node + g

def CrossNodeNghr(node, g):
    nghrNode = g if node > g else g+1
    nghrG = node if nghrNode > node else node-1
    return nghrNode, nghrG


def CrossNodeRouter(n1, n2):
    return (n2 if n1 > n2 else n2-1) % gpus_per_node

# def IBToUse(n1, n2):
#     g1 = n2 if n1 > n2 else n2-1
#     g2 = n1 if n2 > n1 else n1-1
#     # invariant: CrossNodeNghr(n1, g1) == (n2, g2) and vice versa
#     return g1, g2

with SCCLProgram("3step_all_to_all", NDv4(num_nodes, gpus_per_node)):
    for n1 in num_nodes:
        for g1 in gpus_per_node:
            for n2 in num_nodes:
                for g2 in gpus_per_node:
                    r1 = RankFromNodeGpuPair(n1, g1)
                    r2 = RankFromNodeGpuPair(n2, g2)
                    c = Rank(r1).input(r2)

                    if (n1 != n2): 
                        # general case: 
                        # (n1, g1) --gather--> (n1, h1) --IB--> (n2, h2) --scatter--> (n2, g2)  
                        h1 = CrossNodeRouter(n1, n2)
                        h2 = CrossNodeRouter(n2, n1)
                        
                        # Local Gather. 
                        # All chunks destined to n2 should be sent together
                        # in case of h1 = g1, do a copy to scratch buffer
                        c = c.send(RankFromNodeGpuPair(n1, h1))

                        # IB Send. All chunks from all local nodes destine to n2 should be sent together   
                        c = c.send(RankFromNodeGpuPair(n2, h2))

                        # Local Scatter
                        c = c.send(RankFromNodeGpuPair(n2, g2))  
                    elif (g1 != g2):
                        c = c.send(r2) # this should be coalesced with the first send above
                    else:
                        # copy input to output. I dont know how to do that in SCCLang


with SCCLProgram("2step_all_to_all", NDv4(num_nodes, gpus_per_node)):
    for n1 in num_nodes:
        for g1 in gpus_per_node:
            for n2 in num_nodes:
                for g2 in gpus_per_node:
                    r1 = RankFromNodeGpuPair(n1, g1)
                    r2 = RankFromNodeGpuPair(n2, g2)
                    c = Rank(r1).input(r2)

                    if (n1 != n2): # general case 

                        # Local Gather. 
                        # All chunks destined to (n2, g2) should be sent together
                        # in case of h1 = g1, do a copy to scratch buffer
                        c = c.send(RankFromNodeGpuPair(n1, g2))

                        # IB Send
                        c = c.send(RankFromNodeGpuPair(n2, g2))  
                    elif (g1 != g2):
                        c = c.send(r2) # this should be coalesced with the first send above
                    else:
                        # copy input to output. I dont know how to do that in SCCLang
