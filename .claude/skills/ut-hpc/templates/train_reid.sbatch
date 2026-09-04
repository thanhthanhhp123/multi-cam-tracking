#!/bin/bash
# M6 - fine-tune OSNet Re-ID on the project's OWN lab footage.
# NOT YET RUN END-TO-END on the cluster (as of 2026-09-04).
#
# This is NOT part of the main pipeline. M3 ships pretrained OSNet as-is (see CLAUDE.md #9):
# torchreid's checkpoints were already trained on Market-1501/MSMT17, so fine-tuning on those
# same benchmarks re-learns data the model has seen. Run this ONLY after measuring a real
# domain gap on lab data, and report it as an ablation: with vs. without fine-tuning.
#
# STILL MISSING for lab data: torchreid needs the lab identities registered as a custom
# dataset (torchreid.data.register_image_dataset) inside train_reid.py. Do that first --
# SOURCES below is a torchreid dataset NAME, not a path.
#
# Prerequisites, all on the HEAD NODE (compute nodes are offline):
#   1. ~/mct/env/bin/pip install torchreid gdown
#   2. Lab crops under ~/mct/data/reid/lab/ in torchreid layout, uploaded with rsync.
#   3. ~/mct/env/bin/python ~/mct/jobs/fetch_pretrained.py   (caches OSNet ImageNet weights)
#
#SBATCH --job-name=mct-reid
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/mct/logs/slurm-%j.out

set -euo pipefail
ROOT="$HOME/mct"
PY="$ROOT/env/bin/python"

SOURCES="${SOURCES:-lab}"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-64}"

export HF_HUB_OFFLINE=1 TORCH_HOME="$ROOT/.torch"

echo "node=$(hostname) job=$SLURM_JOB_ID sources=$SOURCES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -d "$ROOT/data/reid" ] || { echo "FATAL: thieu $ROOT/data/reid"; exit 1; }

"$PY" "$ROOT/jobs/train_reid.py" \
  --data-root "$ROOT/data/reid" \
  --sources "$SOURCES" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --save-dir "$ROOT/runs/reid" \
  --onnx-out "$ROOT/models/reid/osnet_x1_0_lab.onnx"

ls -lh "$ROOT/models/reid/"
echo "=== keo ve may dev: hpc.sh fetch mct/models/reid/osnet_x1_0_lab.onnx models/reid/osnet_x1_0_lab.onnx ==="
