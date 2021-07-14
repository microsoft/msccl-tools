# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sccl.autosynth as _autosynth

def autosynth(logging=False, torch_distributed_launch=False):
    _autosynth.init(logging=logging, torch_distributed_launch_hack=torch_distributed_launch)
