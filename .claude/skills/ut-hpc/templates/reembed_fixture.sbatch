#!/bin/bash
# Re-extract Re-ID embeddings for the DeepStream fixture, twice: once from the DETECTOR
# boxes and once from the ground-truth boxes, both with the same ONNX Runtime extractor.
#
# Why: session 11 found appearance is the bottleneck (appearance-only ceiling F1 0.181), but
# the two fixtures on hand differ in TWO variables at once (box source AND extractor). These
# two outputs differ in exactly one variable each, so the effect can be attributed:
#
#   gtbox  vs detbox            -> effect of box quality      (same extractor, same tracks)
#   detbox vs the original DeepStream fixture -> effect of extraction path (same boxes, same tracks)
#
# CPU only, no GPU. Needs cv2 + onnxruntime -> ~/mct/venv-reid (NOT venv-test).
#
# Prerequisites, all already on disk (compute nodes have no internet):
#   ~/mct/data/wildtrack/Image_subsets/C1..C7
#   ~/mct/data/fixtures/ds_wildtrack_7cam.jsonl
#   ~/mct/models/reid/osnet_x1_0_msdc_dg.onnx (+ .onnx.data next to it)
#
#SBATCH --job-name=mct-reembed
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/mct/logs/slurm-%j.out

set -euo pipefail

REPO="${MCT_REPO:-$HOME/mct/repo}"
PY="${MCT_PY:-$HOME/mct/venv-reid/bin/python}"
WILDTRACK="${MCT_WILDTRACK:-$HOME/mct/data/wildtrack}"
FIXTURES="${MCT_FIXTURES:-$HOME/mct/data/fixtures}"
REID_ONNX="${MCT_REID_ONNX:-$HOME/mct/models/reid/osnet_x1_0_msdc_dg.onnx}"
IN="${MCT_IN:-$FIXTURES/ds_wildtrack_7cam.jsonl}"

echo "node=$(hostname)  job=${SLURM_JOB_ID:-?}  cpus=${SLURM_CPUS_PER_TASK:-?}"
[ -f "$IN" ] || { echo "FATAL: thieu $IN"; exit 1; }
[ -f "$REID_ONNX" ] || { echo "FATAL: thieu $REID_ONNX"; exit 1; }
[ -d "$WILDTRACK/Image_subsets" ] || { echo "FATAL: thieu anh goc"; exit 1; }

cd "$REPO"
for BOXES in fixture gt; do
    case "$BOXES" in
        fixture) SUFFIX=detbox ;;
        gt)      SUFFIX=gtbox ;;
    esac
    echo "=== boxes=$BOXES -> $SUFFIX ==="
    PYTHONIOENCODING=utf-8 PYTHONPATH=src "$PY" -m tools.reembed_fixture \
        --fixture "$IN" \
        --wildtrack-dir "$WILDTRACK" \
        --reid-onnx "$REID_ONNX" \
        --boxes "$BOXES" \
        --out "$FIXTURES/ds_wildtrack_7cam_onnx_${SUFFIX}.jsonl"
done

echo "--- ket qua ---"
ls -la "$FIXTURES" | grep onnx_
