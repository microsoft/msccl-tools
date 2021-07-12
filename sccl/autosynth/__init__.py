# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.topologies.nvidia import nvlink_only
import subprocess
import re

def detect_node_type():
    node_type = _detect_nvidia_node_type()
    if node_type != None:
        return node_type

def _detect_nvidia_node_type():
    try:
        smi_topo = subprocess.check_output(['nvidia-smi', 'topo', '-m']).decode("utf-8")
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError:
        return 'unknown'

    nvlink_topo = nvlink_only(smi_topo)

    if nvlink_topo.num_nodes == 8: # DGX-1 and DGX A100 like nodes
        if _is_one_host_ib_dgx1():
            return 'one_host_ib_dgx1'

def _is_one_host_ib_dgx1(smi_topo):
    ib_host = re.findall('^mlx\\d_\\d(\s+NODE)*\s+X(\s+NODE)*&', smi_topo, re.MULTILINE)
    ib_any = re.findall('^mlx\\d_\\d.*&', smi_topo, re.MULTILINE)
    return len(ib_host) == 1 and len(ib_any) == 1
