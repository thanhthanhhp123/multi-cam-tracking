#!/bin/bash
# Smoke test: proves the conda env in $HOME reaches a GPU on a compute node.
# Run this once after `hpc.sh setup`, before any real training job.
#SBATCH --job-name=mct-smoke
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/%u/mct/logs/slurm-%j.out

set -euo pipefail
PY="${MCT_PY:-$HOME/mct/env/bin/python}"

echo "node=$(hostname)  job=$SLURM_JOB_ID  gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Compute nodes have no internet: keep every library offline.
export ULTRALYTICS_OFFLINE=1 HF_HUB_OFFLINE=1 TORCH_HOME="$HOME/mct/.torch"

"$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
a = torch.randn(4096, 4096, device="cuda")
(a @ a).sum().item()
print("matmul ok, vram_total_GB=%.1f" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
try:
    import ultralytics
    print("ultralytics", ultralytics.__version__)
except ImportError:
    print("ultralytics: NOT INSTALLED")
PYEOF
