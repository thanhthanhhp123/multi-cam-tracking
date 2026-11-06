#!/bin/bash
# Gan nhan TRUOC cho CVAT (M6) tren instance vast.ai — tools/prelabel_cvat.py, docs/m6/README.md §5.
# KHONG can DeepStream: image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime.
#
# Tu may dev, goc repo (3 luong tar song song, moi luong mot camera — duong VN -> may thue
# RTT cao, mot luong TCP khong dung het bang thong):
#   tar cf - src configs/lab docker/vast_prelabel.sh | ssh vast-gpu 'mkdir -p /workspace/mct-repo && tar xf - -C /workspace/mct-repo'
#   for c in cam01 cam02 cam03; do tar cf - data/lab/*/$c.mp4 | ssh vast-gpu 'tar xf - -C /workspace/mct-repo' & done; wait
# Tren instance:
#   bash /workspace/mct-repo/docker/vast_prelabel.sh setup
#   nohup bash /workspace/mct-repo/docker/vast_prelabel.sh run > /workspace/prelabel.log 2>&1 < /dev/null &
# Ve may dev:
#   ssh vast-gpu 'cd /workspace/mct-repo && tar cf - data/prelabel logs' | tar xf -
# NHO HUY MAY (tinh tien theo gio) — video nguoi that khong duoc nam lai tren may thue
# (docs/m6/phieu-dong-thuan.md muc 3).
set -euo pipefail
REPO=/workspace/mct-repo
cd "$REPO"

MODEL=yolo11x.pt   # KHAC YOLO11s 640 cua pipeline dang cham — xem docstring cua tool
IMGSZ=1280         # nguoi o cuoi hanh lang chi cao vai chuc pixel
FOOT_CAM01=250     # quy uoc chu thich: chan duoi y ~ 270 tren cam01, lay du ra mot chut

case "${1:?buoc: setup | run}" in
setup)
  # opencv-python (ultralytics keo theo) can libGL; image runtime cua pytorch khong co.
  apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 > /dev/null
  pip install -q "ultralytics==8.4.7" lap python-dotenv   # dotenv: common.logging -> common.config
  python -c "import torch, cv2, ultralytics; print(torch.cuda.get_device_name(0), 'cv2', cv2.__version__, 'ultralytics', ultralytics.__version__)"
  # Tai trong so MOT lan, truoc khi 3 tien trinh cung tai vao mot file.
  python -c "from ultralytics import YOLO; YOLO('$MODEL')"
  ls data/lab/*/cam0?.mp4 | wc -l
  ;;
run)
  mkdir -p logs data/prelabel
  nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks_throttle_reasons.active,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 10 > logs/prelabel_gpu.csv 2>/dev/null &
  GPU_LOG_PID=$!
  # Moi camera mot tien trinh: mot luong YOLO batch 1 khong lap day GPU, va giai ma 1080p chay CPU.
  pids=()
  for cam in cam01 cam02 cam03; do
    extra=()
    [ "$cam" = cam01 ] && extra=(--min-foot-y "cam01=$FOOT_CAM01")
    PYTHONPATH=src python -m tools.prelabel_cvat data/lab/*/"$cam".mp4 --out-dir data/prelabel \
      --model "$MODEL" --imgsz "$IMGSZ" "${extra[@]}" > "logs/prelabel_$cam.log" 2>&1 &
    pids+=($!)
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
  kill "$GPU_LOG_PID" 2>/dev/null || true
  echo "tien trinh loi: $failed/3"
  ls data/prelabel/*/*.xml | wc -l
  tail -n 20 logs/prelabel_cam0?.log
  ;;
*)
  echo "buoc khong hop le: $1" >&2; exit 2 ;;
esac
