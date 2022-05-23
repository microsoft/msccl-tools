## Synthesizing Algorithms

MSCCL can synthesize algorithms for a given *topology* that implements a given *collective* in a given number of steps, bandwidth usage, memory limits, etc. These additional parameters are called the *instance*.

MSCCL groups its solver strategies under the `msccl solve` subcommand. For example, to synthesize a specific `instance` of an Allgather algorithm for the [NVIDIA DGX-1](https://www.nvidia.com/en-us/data-center/dgx-1/) that completes in 4 steps:
```
$ msccl solve instance DGX1 Allgather --steps 4
Solving instance steps=4... synthesized! (0.7s)
Wrote to Allgather.n8-DGX1-steps4.msccl.json
```
The instance is satisfiable and `msccl` saves it to a file.

Four steps is not necessarily the least number of steps required. To find the least steps:
```
$ msccl solve least-steps DGX1 Allgather
Algorithms need at least 2 steps.
Solving instance steps=2... synthesized! (0.2s)
Wrote to Allgather.n8-DGX1-steps2.msccl.json
```
The `least-steps` strategy statically determines that any Allgather in a DGX-1 requires at least 2 steps and starting from that finds the smallest satisfiable number of steps.

While this two step algorithm is a latency-optimal one, there may be other algorithms that achieve higher bandwidth. The `pareto-optimal` strategy searches through different latency-bandwidth tradeoffs:
```
$ msccl solve pareto-optimal DGX1 Allgather
Algorithms need at least 2 steps.
Algorithms need at least 7/6 rounds per chunk.
Solving instance steps=2... synthesized! (0.5s)
Solving instance steps=2,rounds=3,chunks=2... synthesized! (0.9s)
Solving instance steps=2,rounds=4,chunks=3... unsatisfiable. (1.1s)
Assuming 2 step algorithms need at least 4/3 rounds per chunk.
Solving instance steps=3,rounds=4,chunks=3... synthesized! (2.9s)
Solving instance steps=3,rounds=5,chunks=4... synthesized! (6.5s)
Solving instance steps=3,rounds=6,chunks=5... synthesized! (44.0s)
Solving instance steps=3,rounds=7,chunks=6... synthesized! (56.1s)
Bandwidth optimal algorithm found!
Found 2 Pareto optimal algorithms. Pruned 4 non-optimal algorithms.
Wrote to Allgather.n8-DGX1-steps2.rounds3.chunks2.msccl.json
Wrote to Allgather.n8-DGX1-steps3.rounds7.chunks6.msccl.json
```

## Collectives

MSCCL includes a number of built in common collectives.

| Collective | Arguments | Description | Kind |
| - | - | - | - |
| Broadcast | `--root N` | Send data from root to all nodes. | NC |
| Reduce | `--root N` | Combine data from all nodes to root. | CR |
| Scatter | `--root N` | Send slices of data from root to all nodes. | NC |
| Gather | `--root N` | Send slices of data from all nodes to root. | NC |
| Allgather | | Send slices of data from all nodes to all nodes. | NC |
| Allreduce | | Combine data from all nodes to all nodes. | CNR |
| Alltoall | | Transpose data between all nodes. | NC |
| ReduceScatter | | Combine slices of data to all nodes. | CR |
| Scan | | Combine partial prefixes of data to all nodes in sequence. | CNR |
| MultirootBroadcast | `--roots N [N ...]` | Like Broadcast, but set of nodes have slices of input. | NC |
| MultirootScatter | `--roots N [N ...]` | Like Scatter, but set of nodes have slices of input. | NC |
| MultirootGather | `--roots N [N ...]` | Like Gather, but output is sent in slices to a set of nodes. | NC |
| custom | `--collective-file` | Arbitrary collective serialized by the user. | ? |

Custom collectives may be defined by instantiating the `Collective` class, which is easiest through the `build_collective` function. For example, a send from rank 2 to rank 7 in an 8 node topology can be defined and saved with:
```
from msccl.collectives import build_collective
from msccl.serialization import save_msccl_object

precondition = lambda r, c: r == 2
postcondition = lambda r, c: r == 7
coll = build_collective('Send', 8, 1, precondition, postcondition)
save_msccl_object(coll, 'send.json')
```

The *kind* of the collective determines support for some features of MSCCL:
- **NC** are non-combining collectives, and are always supported.
- **CR** are combining collectives that have a non-combining dual collective, and are supported through a reduction.
- **CNR** are combining collectives with no dual, which may not always be supported.

Currently the rounds per chunk analysis described below can not support CNR collectives.

## Steps and Rounds

MSCCL uses two related concepts, *steps and rounds*, to talk about the running time of algorithms. *Steps* is how many sequential sets of sends the algorithm consists of, where all sends inside a step execute in parallel. The number of sends between two nodes in a single step is limited by the bandwidth available in the topology. However, a step may consist of multiple *rounds*, which acts as a multiplier for all links in the topology during that step.

How much data a single round corresponds to depends on what is the actual size of a chunk at runtime, and how many chunks a collective uses can change (e.g. you can control this directly in the `instance` strategy by setting `--chunks N`). Thus for each collective the total data usage of different algorithms implementing it can be measured with their *rounds per chunk*.

MSCCL provides a standalone analysis to find a lower bound for the *rounds per chunk* required by any instance. For example, to find the least rouds per chunk for an Alltoall in a DGX-1:
```
$ msccl analyze rounds DGX1 Gather
Gather(n=8,root=0) algorithms need at least 7/6 rounds in DGX1 topology.
```
In this case the bound happens to be tight and the `pareto-optimal` strategy would use it to detect that it has found a bandwidth optimal algorithm.

## Distributed Algorithms

MSCCL provides routines to synthesize algorithms for distributed topologies under the `msccl distribute` subcommand. These work by using algorithms for a local collective and stitcing instances of it together to create a distributed one.

**Alltoall from Gather and Scatter:** `alltoall-gather-scatter` combines a Gather and a Scatter algorithm with a transpose step in the middle to form a distributed Alltoall algorithm. For example, an Alltoall algorithm for a cluster of 4 DGX-1 machines can be created with:
```
msccl solve least-steps DGX1 Gather -o gather.json
msccl solve least-steps DGX1 Scatter -o scatter.json --root 1
msccl distribute alltoall-gather-scatter gather.json scatter.json --copies 4 -o alltoall.json
```
This distributor works with any Gather and Scatter algorithm, as long as their roots have a direct connection in the topology. MSCCL also provides multi-root versions of Gather and Scatter that can be substituted here.
