#!/bin/bash
# Pack WildTrack image sequences into H.264 videos so the DeepStream pipeline can ingest
# them (it takes video/RTSP, not a directory of PNGs). CPU only -- no --gres=gpu here.
#
# Why this job exists: the `max_cost` threshold in configs/demo/wildtrack.mct.yaml was tuned
# on embeddings from ONNX Runtime on CPU, while the real pipeline extracts them with TensorRT
# inside nvtracker. Running the SAME dataset through DeepStream is the only way to tell whether
# that threshold transfers. See docs/worklog/2026-09-04-9-*.md ("Vướng mắc").
#
# Prerequisites (all already on disk, nothing here touches the network):
#   1. ~/mct/data/wildtrack/{annotations_positions,Image_subsets/C1..C7}
#   2. ~/mct/repo/src  -- push with:  tar czf - src | ssh ut-hpc 'tar xzf - -C ~/mct/repo'
#   3. ffmpeg with libx264 (verified on the head node 2026-09-04: 4.4.2-0ubuntu0.22.04.1)
#
# Output: ~/mct/data/wildtrack_video/cam0N.mp4 + index.json, then rsync to vast-gpu.
#
# main-cpu, not main-gpu: encoding needs no GPU, and occupying a GPU node for it would be
# rude on a shared cluster.
#
# NOTE (measured 2026-09-04): compute nodes do NOT carry identical images. ffmpeg 4.4.2 is
# everywhere, but ctit085 and hpc-node09 ship it WITHOUT libx264 (jobs 581854, 581864 died on
# them), while ctit086/ctit087 and the head node have it. So "it worked last time" proves
# nothing -- SLURM hands you a different node. The guard below is what turns that into a loud
# failure on line one instead of a confusing one halfway through.
#
#SBATCH --job-name=mct-wtvideo
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/mct/logs/slurm-%j.out

set -euo pipefail

REPO="${MCT_REPO:-$HOME/mct/repo}"
PY="${MCT_PY:-$HOME/mct/venv-test/bin/python}"
WILDTRACK="${MCT_WILDTRACK:-$HOME/mct/data/wildtrack}"
OUT="${MCT_VIDEO_OUT:-$HOME/mct/data/wildtrack_video}"

echo "node=$(hostname)  job=${SLURM_JOB_ID:-?}  cpus=${SLURM_CPUS_PER_TASK:-?}"

# Compute nodes may carry a different image than the head node -- fail loudly and early
# rather than halfway through the seventh camera.
command -v ffmpeg >/dev/null || { echo "FATAL: khong co ffmpeg tren node nay"; exit 1; }
command -v ffprobe >/dev/null || { echo "FATAL: khong co ffprobe tren node nay"; exit 1; }
ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 \
    || { echo "FATAL: ffmpeg tren node nay khong co libx264"; exit 1; }
ffmpeg -version | head -1

[ -d "$WILDTRACK/Image_subsets" ] || { echo "FATAL: thieu $WILDTRACK/Image_subsets"; exit 1; }
[ -d "$REPO/src" ] || { echo "FATAL: chua day repo len $REPO"; exit 1; }

mkdir -p "$OUT"
cd "$REPO"
PYTHONPATH=src "$PY" -m tools.wildtrack_to_video \
    --wildtrack-dir "$WILDTRACK" \
    --out-dir "$OUT" \
    "$@"

echo "--- ket qua ---"
ls -la "$OUT"
du -sh "$OUT"
