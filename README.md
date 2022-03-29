# SCCL

SCCL is a programmable GPU communication library that offers synthesis tools and a programming language, SCCLang, for
building collective algorithms tailored to a particular hardware and workload.

## Installation

### Python package and tool

To install either clone this repo and run "`pip install .`" or run:
```
pip install git+https://github.com/microsoft/sccl.git
```
This installs the Python package and the `sccl` command line tool.

To enable Bash completion for the `sccl` tool:
```
echo 'eval "$(register-python-argcomplete sccl)"' >> ~/.bashrc
```

### Runtime

SCCL's algorithms run in [a modified version of NCCL that includes an interpreter](https://github.com/microsoft/msccl),
which is API compatible with NCCL and is installed as normal. See https://github.com/microsoft/msccl for instructions.

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

## Usage

The SCCL Python package ships with a registry of synthesis strategies and hand optimized algorithms. These can be loaded
into [the runtime](https://github.com/parasailteam/msccl) through the `sccl.init` function, which must be called before
the application creates its NCCL communicator. For PyTorch this means before `torch.distributed` is initialized.

The following snippet requests `sccl.init` to provide an Alltoall algorithm in a configuration of 2 Azure NDv2 machines:
```
import sccl
sccl.init('ndv2', 2, (sccl.Collective.alltoall, ('1MB')))
```
The call will finds an algorithm provider that can create an Alltoall algorithm that is expected to be good with 1MB of
data. That will call a synthesis routine that writes the algorithm to disk. `sccl.init` will then pass a configuration
file pointing to this algorithm to the runtime through environment variables.

See [the examples](examples/sccl_init.py) for more on `sccl.init` usage.

Refer to the next section on availability of algorithms with `sccl.init`.

### Note on Azure NDv2

Azure NDv2 does not expose the true PCIe topology of the machines to the VM and worse, does not assign PCIe devices
consistently to the virtual paths in the VM. As SCCL is generating topology-aware algorithms, this device ordering must
be fixed. The [sccl_ndv2_launcher.sh](sccl/autosynth/sccl_ndv2_launcher.sh) script can be used to fix this problem. The
script solves the automorphisms from the local VM's NVLink topology to the reference topology and selects one of the 4
automorphisms based on measured placement of the Infiniband card such that GPU 0 is close to the NIC. A tool called
[inspector-topo](https://github.com/microsoft/inspector-topo) needs to be available for the latter step.

## Available Algorithms

SCCL's built-in algorithm providers currently includes an efficient Alltoall algorithm for Azure NDv2 nodes. Stay tuned
for more algorithms coming soon!

https://github.com/parasailteam/sccl-presynth offers additional algorithms that have been pre-synthesized for fixed
configurations. To enable them install the package and import it before the call to `sccl.init`.

## Synthesis

SCCL started out as a synthesizer for collective algorithms, and has since expanded to cover a broader range of
programmability. See [this readme](SYNTHESIS.md) for using SCCL as a synthesizer.

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
