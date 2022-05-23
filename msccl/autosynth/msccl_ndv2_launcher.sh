#!/bin/bash
python -c "import msccl; msccl.ndv2_perm()"
order=/var/lock/msccl_autosynth_inspector_topo.lock
if [ -f "$order" ]; then
    export CUDA_VISIBLE_DEVICES=$(</var/lock/msccl_autosynth_inspector_topo.lock)
    echo "Set CUDA_VISIBLE_DEVICES to: "$CUDA_VISIBLE_DEVICES
fi
$@