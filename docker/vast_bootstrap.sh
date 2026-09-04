#!/bin/bash
# Dung moi truong DeepStream tren instance vast.ai, chay TREN chinh instance do.
#
# VI SAO KHONG DUNG DOCKER: instance vast.ai ban than no la mot container va khong co
# Docker daemon ben trong (`docker: command not found`) — do la gioi han cua nen tang,
# khong phai cua docker/deepstream.Dockerfile. Script nay chay dung chuoi lenh RUN cua
# Dockerfile do mot cach native. Doi Dockerfile thi phai doi ca file nay.
# Xem docs/worklog/2026-09-04-2-* quyet dinh 5.
#
# Dung:
#   scp docker/vast_bootstrap.sh vast-gpu:/workspace/
#   ssh vast-gpu 'bash /workspace/vast_bootstrap.sh 2>&1 | tail -40'
#
# Idempotent: chay lai tren instance da dung xong thi bo qua cac buoc da co.

set -euo pipefail

DS_ROOT=/opt/nvidia/deepstream/deepstream
REPO=/workspace/mct-repo
YOLO_DIR=/workspace/DeepStream-Yolo
PYDS_URL="https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_x86_64.whl"
CUDA_VER="${CUDA_VER:-12.6}"

buoc() { echo; echo "=== $* ==="; }

buoc "0. May nay la gi"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 --version
echo "DeepStream: $(cat "$DS_ROOT/version" 2>/dev/null || echo '?')"

buoc "1. user_additional_install.sh"
if [ -f /workspace/.done-additional ]; then
  echo "da chay truoc do, bo qua"
else
  (cd "$DS_ROOT" && ./user_additional_install.sh)
  touch /workspace/.done-additional
fi

buoc "2. pyds 1.2.0 (cp310 — KHONG dung cp312, do la ban cho DeepStream 8.0/Ubuntu 24.04)"
python3 -c "import pyds" 2>/dev/null && echo "pyds da co" || pip install --no-cache-dir "$PYDS_URL"
python3 -c "import pyds; print('pyds', pyds.__file__)"

buoc "3. DeepStream-Yolo (parser bbox cho YOLO11 — DeepStream goc khong co)"
[ -d "$YOLO_DIR" ] || git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo.git "$YOLO_DIR"

buoc "4. deps python cho buoc export ONNX"
python3 -c "import ultralytics, onnx" 2>/dev/null && echo "da co" \
  || pip install --no-cache-dir ultralytics onnx onnxslim onnxscript onnxruntime

buoc "5. build libnvdsinfer_custom_impl_Yolo.so (CUDA_VER=$CUDA_VER)"
LIB="$YOLO_DIR/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"
if [ -f "$LIB" ]; then
  echo "da build: $LIB"
else
  CUDA_VER="$CUDA_VER" make -C "$YOLO_DIR/nvdsinfer_custom_impl_Yolo" -j"$(nproc)"
fi
ls -la "$LIB"

buoc "6. repo + third_party symlink"
[ -d "$REPO/src" ] || { echo "FATAL: chua day repo len $REPO (dung rsync/tar tu may dev)"; exit 1; }
mkdir -p "$REPO/third_party" "$REPO/models/detector" "$REPO/models/reid"
ln -sfn "$YOLO_DIR" "$REPO/third_party/DeepStream-Yolo"
(cd "$REPO" && pip install --no-cache-dir -e . >/dev/null && echo "pip install -e . ok")

buoc "7. redis-server (de chay duong --publish + ghi fixture)"
command -v redis-server >/dev/null && echo "da co" || { apt-get update -qq && apt-get install -y -qq redis-server; }
redis-server --version

buoc "8. kiem tra import"
(cd "$REPO" && python3 -c "
import pyds, ds_pipeline, common.schema
from ds_pipeline import reid_meta
print('import ok — pyds + ds_pipeline + reid_meta')
print('NVDS_TRACKER_OBJ_REID_META co trong pyds:',
      hasattr(pyds.NvDsMetaType, 'NVDS_TRACKER_OBJ_REID_META'))
")

buoc "9. trong so"
ls -la "$REPO/models/detector/" "$REPO/models/reid/" 2>/dev/null || true
echo
echo "Neu models/ trong: day tu may dev bang"
echo "  rsync -av models/ vast-gpu:$REPO/models/"

buoc "XONG"
echo "Buoc tiep: export yolo11s.pt -> ONNX neu chua co, roi chay checklist M3."
