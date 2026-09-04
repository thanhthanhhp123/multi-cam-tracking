#!/bin/bash
# M6 - fine-tune YOLO on the project's OWN lab footage, then export ONNX for DeepStream.
# NOT YET RUN END-TO-END on the cluster (as of 2026-09-04): treat the first run as a test.
#
# This is NOT part of the main pipeline. M2 ships the pretrained COCO weights as-is
# (see CLAUDE.md #9). Only run this once a domain gap has actually been MEASURED on lab
# data, and report it as an ablation: with vs. without fine-tuning.
#
# Prerequisite: labelled lab data already on disk in Ultralytics layout, uploaded from the
# dev machine (compute nodes have no internet):
#   rsync -av data/lab/detect/ ut-hpc:mct/data/lab/detect/
#
#SBATCH --job-name=mct-yolo
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/mct/logs/slurm-%j.out

set -euo pipefail
ROOT="$HOME/mct"
PY="$ROOT/env/bin/python"

MODEL="${MODEL:-$ROOT/models/yolo11s.pt}"      # staged on the head node beforehand
DATA="${DATA:-$ROOT/data/lab/detect/lab.yaml}"
EPOCHS="${EPOCHS:-50}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-32}"
NAME="${NAME:-yolo11s_lab}"

export ULTRALYTICS_OFFLINE=1 HF_HUB_OFFLINE=1 TORCH_HOME="$ROOT/.torch"
export YOLO_CONFIG_DIR="$ROOT/.ultralytics"

echo "node=$(hostname) job=$SLURM_JOB_ID model=$MODEL data=$DATA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f "$MODEL" ] || { echo "FATAL: thieu $MODEL (tai tren head node truoc)"; exit 1; }
[ -f "$DATA" ]  || { echo "FATAL: thieu $DATA (rsync du lieu lab len cum truoc)"; exit 1; }

"$PY" -m ultralytics.cfg train \
  model="$MODEL" data="$DATA" epochs="$EPOCHS" imgsz="$IMGSZ" batch="$BATCH" \
  device=0 workers=8 project="$ROOT/runs/detect" name="$NAME" exist_ok=True

BEST="$ROOT/runs/detect/$NAME/weights/best.pt"
echo "=== export ONNX ==="
"$PY" -m ultralytics.cfg export model="$BEST" format=onnx opset=12 simplify=True dynamic=False imgsz="$IMGSZ"

mkdir -p "$ROOT/models/detector"
cp "$BEST" "$ROOT/models/detector/$NAME.pt"
cp "${BEST%.pt}.onnx" "$ROOT/models/detector/$NAME.onnx"
ls -lh "$ROOT/models/detector/"
echo "=== keo ve may dev: hpc.sh fetch mct/models/detector/$NAME.onnx models/detector/$NAME.onnx ==="
