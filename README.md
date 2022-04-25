# SCCL

SCCL is a tool stack for programmable communication on GPUs. Algorithms created with SCCL can:
- Implement either MPI-style collectives like Allreduce, or any application specific communication pattern.
- Target specific hardware and interconnect topologies, unlocking their full potential.
- Optimize for the data sizes in your application, making the best tradeoff between latency and bandwidth utilization.

SCCL ships with algorithms targeting various Azure multi-GPU VM types. See the [Available Algorithms section](#available-algorithms) to find out what is currently available.

SCCL has two ways of creating new algorithms:
1. MSCCLang, a high-level DSL that talks about communication in an intuitive chunk-oriented form. See the [MSCCLang
section](#mscclang) for how to get started.
2. Synthesis, which automatically solves optimal algorithms for a given hardware topology. Making synthesis general
enough for common use cases is an on-going research project See [the synthesis readme](SYNTHESIS.md) for an
introduction.

## When is SCCL likely to help

**Built-in algorithms:** If you wish to just use SCCL's built-in algorithms, take stock of the collectives used in your model and check if those are available for your Azure VM type (see the [Available Algorithms section](#available-algorithms)). Further, you should make sure that the data sizes your model operates with are covered by the sizes the algorithms are registered for. As a rule-of-thumb, model- and pipeline parallelism result in small to medium data sizes, while data parallelism results in large data sizes for communication.

If your use-case seems to be at least partially covered by the available algorithms, there is a good chance SCCL will provide a good speedup.

**New algorithms:** If you are willing to dive into developing new algorithms with for example MSCCLang, here are some scenarios where you may find great improvements:
- *Your model uses small or medium sized data, but NCCL only optimizes for large data sizes.* NCCL's ring based algorithms can be great for large data, but they require many steps to complete and thus pay fixed latency costs many times. Algorithms with "shallower" communication paths may be better.
- *Your hardware configuration is atypical.* For example, if your NIC placement is closer to some GPUs than others you may find a way to route data in a more optimal way.
- *You require non-mainstream collectives.* Alltoall is a well-known collective that historically has not been required
  for ML workloads, but is finding some emerging use cases.

## Usage

The SCCL Python package ships with a registry of synthesis strategies and hand optimized algorithms. These can be loaded
into [the runtime](https://github.com/parasailteam/msccl) through the `sccl.init` function, which must be called before
the application creates its NCCL communicator. For PyTorch this means before `torch.distributed` is initialized.

The following snippet requests `sccl.init` to provide an Alltoall algorithm in a configuration of 2 Azure NDv2 machines:
```
import sccl
sccl.init('ndv2', 2, (sccl.Collective.alltoall, ('1MB')))
```
This will find an algorithm provider that can create an Alltoall algorithm that is expected to be good with 1MB of data.
That will call a synthesis routine that writes the algorithm to disk. `sccl.init` will then pass a configuration file
pointing to this algorithm to the runtime through environment variables.

See [the examples](examples/sccl_init.py) for more on `sccl.init` usage.

## Available Algorithms

SCCL's built-in algorithms are registered for combinations of hardware configuration and size of input data where we
have benchmarked them to provide speedup over NCCL. To list the algorithms currently in SCCL's built-in registry, run
`sccl plans list` on the command line. This will print out the following table (on 4/22/2022):

| Machine   | Collective   | # machines   | From   | To       | Protocol   |   Priority | Plan name                           |
|-----------|--------------|--------------|--------|----------|------------|------------|-------------------------------------|
| ndv2      | alltoall     | >=2          | 1 MB   | infinity | Simple     |          0 | call synthesize_ndv2_relay_alltoall |
| ndv4      | allreduce    | 1            | 256 KB | 20 MB    | LL128      |          0 | run ndv4_ring_allreduce             |
| ndv4      | alltoall     | 8,16,32,64   | 1 MB   | 32 MB    | LL128      |          0 | run ndv4_alltoall_hierarchical      |
| ndv4      | alltoall     | 8,16,32      | 32 MB  | infinity | Simple     |          0 | run ndv4_alltoall_hierarchical      |
| ndv4      | alltoall     | 64           | 32 MB  | infinity | Simple     |          0 | run ndv4_alltoall_three_step        |

Each line lists an algorithm registration and the conditions under which it is triggered. For example, the
`ndv4_alltoall_hierarchical` algorithm will be used with NCCL's lower latency LL128 protocol when:
- the user has called Alltoall,
- there are 1, 2, 4 or 8 Azure NDv4 machines, and
- the data size is from 1 MB to 32 MB.

The repository [parasailteam/sccl-presynth](https://github.com/parasailteam/sccl-presynth) repository offers additional algorithms that have been
pre-synthesized for fixed configurations. To enable them install the package and import it before the call to
`sccl.init`.

## Troubleshooting

If you've integrated SCCL into your model, but aren't seeing a speedup, there are several reasons SCCL's algorithms may not be getting used.

**Algorithms not selected:** `sccl.init` prints diagnostic information indicating which algorithms are selected. 
```
SCCL: Plan for <collective> with sizes from <low> to <high> is load <plan> with <protocol>.
```
When no algorithm exists for some requested collective and size range, you will see a printout like the following:
```
SCCL: No plan found for <collective> with sizes from <low> to <high>. Falling back to NCCL baseline.
```
Please check that 1. an algorithm is available for your request (see the [Available Algorithms
section](#available-algorithms)) and 2. the arguments to `sccl.init` overlap with the ones the algorithm is registered
with.

**Algorithms not loaded:** After `sccl.init` has selected algorithms, it informs the [MSCCL
runtime](https://github.com/microsoft/msccl) which ones to load by setting the appropriate environment variables. To verify that this happens please set `NCCL_DEBUG=INFO` in the environment. For each algorithm you should see:
```
NCCL INFO Parsed SCCL Algorithm <path> successfully.
```
And finally when the algorithms were correctly loaded:
```
NCCL INFO Connected <count> SCCL algorithms
```

**Algorithms not triggered at runtime:** Even when SCCL algorithms are loaded into MSCCL, they are only used at runtime
when the incoming data size matches the size range the algorithms were selected for. Please check that the data sizes in
your model's communication calls are covered by the algorithms you see being selected by `sccl.init`.

`sccl.init` also sets the `NCCL_ALGO` environment variable to include `SCCL`. Please check that this is not being overwritten in your model.

## MSCCLang

MSCCLang is a high-level language for specifying collective communication algorithms in an intuitive chunk-oriented form. The language is available as a Python-integrated DSL.

The language is still under development and lacks comprehensive documentation. For now, please refer to [the pre-print of our upcoming paper](https://arxiv.org/pdf/2201.11840.pdf) and the examples in [examples/scclang](examples/scclang/).

## Synthesis

SCCL started out as a synthesizer for collective algorithms, and general synthesis of collective algorithms is an
on-going research project. See [this readme](SYNTHESIS.md) for using SCCL as a synthesizer.

## Installation

### Python Package Installation

To install either clone this repo and run "`pip install .`" or run:
```
pip install git+https://github.com/microsoft/sccl.git
```

Installing the SCCL Python package also installs the `sccl` command line tool. To enable Bash completion for the `sccl`
tool:
```
echo 'eval "$(register-python-argcomplete sccl)"' >> ~/.bashrc
```

### Runtime Installation

SCCL's algorithms are executed by the [Microsoft Collective Communication Library
(MSCCL)](https://github.com/microsoft/msccl), which is API compatible with NCCL. See https://github.com/microsoft/msccl
for instructions.

To use SCCL with PyTorch, the built in NCCL submodule has to be replaced with SCCL's version. Additionally, to expose
the new native Alltoall support that SCCL adds, PyTorch's `torch.distributed` package can optionally be patched. The
following commands perform these steps and install PyTorch with SCCL:
```
git clone https://github.com/pytorch/pytorch.git
cd pytorch    
git checkout tags/v1.9.0 -b v1.9.0_sccl
perl -p -i -e  's/url = https:\/\/github\.com\/NVIDIA\/nccl/url = https:\/\/github\.com\/microsoft\/msccl/g' .gitmodules
git submodule sync third_party/nccl
git submodule update --init --recursive
git submodule update --init --recursive --remote third_party/nccl
git apply third_party/nccl/nccl/patches/nccl.cpp.patch
python setup.py install
```

### Note on Azure NDv2

Azure NDv2 does not expose the true PCIe topology of the machines to the VM and worse, does not assign PCIe devices
consistently to the virtual paths in the VM. As SCCL is generating topology-aware algorithms, this device ordering must
be fixed. The [sccl_ndv2_launcher.sh](sccl/autosynth/sccl_ndv2_launcher.sh) script can be used to fix this problem. The
script solves the automorphisms from the local VM's NVLink topology to the reference topology and selects one of the 4
automorphisms based on measured placement of the Infiniband card such that GPU 0 is close to the NIC. A tool called
[inspector-topo](https://github.com/microsoft/inspector-topo) needs to be available for the latter step.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License
Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For
details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate
the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only
need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact
[opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks
or logos is subject to and must follow [Microsoft's Trademark & Brand
Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft
trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any
use of third-party trademarks or logos are subject to those third-party's policies.
