#!/bin/bash
# M3 - fine-tune OSNet Re-ID. NOT YET RUN END-TO-END on the cluster (as of 2026-09-03).
#
# Prerequisites, all on the HEAD NODE (compute nodes are offline):
#   1. ~/mct/env/bin/pip install torchreid gdown
#   2. Datasets unpacked under ~/mct/data/reid/ in torchreid layout, e.g.
#        ~/mct/data/reid/market1501/Market-1501-v15.09.15/{bounding_box_train,query,...}
#      Market-1501 and MSMT17 need a manual download (licence-gated); copy them up with
#      `scp` or `rsync` from the dev machine if the direct link is dead.
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

SOURCES="${SOURCES:-market1501}"
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
  --onnx-out "$ROOT/models/reid/osnet_x1_0_ft.onnx"

ls -lh "$ROOT/models/reid/"
echo "=== keo ve may dev: hpc.sh fetch mct/models/reid/osnet_x1_0_ft.onnx models/reid/osnet_x1_0_ft.onnx ==="
