#!/bin/bash
python -c "import sccl; sccl.ndv2_perm()"
order=/var/lock/sccl_autosynth_inspector_topo.lock
if [ -f "$order" ]; then
    export CUDA_VISIBLE_DEVICES=$(</var/lock/sccl_autosynth_inspector_topo.lock)
    echo "Setting CUDA_VISIBLE_DEVICES to: "$CUDA_VISIBLE_DEVICES
fi
$@