#!/bin/bash
# M2 - fine-tune YOLO for person detection, then export ONNX for the DeepStream side.
# NOT YET RUN END-TO-END on the cluster (as of 2026-09-03): treat the first run as a test.
#
# Prerequisite (head node, has internet):
#   ssh ut-hpc '~/mct/env/bin/python ~/mct/jobs/prepare_coco_person.py'
# The job below must find the dataset and the pretrained .pt already on disk.
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
DATA="${DATA:-$ROOT/data/coco-person/coco_person.yaml}"
EPOCHS="${EPOCHS:-50}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-32}"
NAME="${NAME:-yolo11s_person}"

export ULTRALYTICS_OFFLINE=1 HF_HUB_OFFLINE=1 TORCH_HOME="$ROOT/.torch"
export YOLO_CONFIG_DIR="$ROOT/.ultralytics"

echo "node=$(hostname) job=$SLURM_JOB_ID model=$MODEL data=$DATA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f "$MODEL" ] || { echo "FATAL: thieu $MODEL (tai tren head node truoc)"; exit 1; }
[ -f "$DATA" ]  || { echo "FATAL: thieu $DATA (chay prepare_coco_person.py truoc)"; exit 1; }

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
