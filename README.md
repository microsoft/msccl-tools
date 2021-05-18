# SCCL

The Synthesized Collective Communication Library is a tool for synthesizing collective algorithms tailored to a particular hardware topology.

## Installation

To install:
```
pip install .
```
This installs the Python package and the `sccl` command line tool.

To enable Bash completion for `sccl`:
```
echo 'eval "$(register-python-argcomplete sccl)"' >> ~/.bashrc
```

## Usage

At its core SCCL answers synthesis queries is there an algorithm for a given *topology* that implements a given *collective* in a given number of steps, bandwidth usage, memory limits, etc. These additional parameters are called the *instance*.

For example, to synthesize an Allgather algorithm for an [NVIDIA DGX-1](https://www.nvidia.com/en-us/data-center/dgx-1/) that completes in 4 steps:
```
$ sccl solve instance DGX1 Allgather --steps 4
Solving instance steps=4... synthesized! (0.7s)
Wrote to Allgather.n8-DGX1-steps4.sccl.json
```
The instance is satisfiable and `sccl` saves it to a file.

Four steps is not necessarily the least number of steps required. To find the least steps:
```
$ sccl solve least-steps DGX1 Allgather
Algorithms need at least 2 steps.
Solving instance steps=2... synthesized! (0.2s)
Wrote to Allgather.n8-DGX1-steps2.sccl.json
```
The `least-steps` strategy statically determines that any Allgather in a DGX-1 requires at least 2 steps and starting from that finds the smallest satisfiable number of steps.

While this two step algorithm is a latency-optimal one, there may be other algorithms that achieve higher bandwidth. The `pareto-optimal` strategy searches through different latency-bandwidth tradeoffs:
```
$ sccl solve pareto-optimal DGX1 Allgather
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
Wrote to Allgather.n8-DGX1-steps2.rounds3.chunks2.sccl.json
Wrote to Allgather.n8-DGX1-steps3.rounds7.chunks6.sccl.json
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
