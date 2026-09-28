#!/bin/bash
# Chay PeopleNet Transformer tren instance vast.ai — phien 30 (docs/worklog/2026-09-28-30-*).
# Chay TREN instance, theo thu tu:
#   0. bash docker/vast_wildtrack.sh nvdec          (CLAUDE.md §11 — LAM DAU TIEN)
#   1. bash docker/vast_bootstrap.sh                (KHONG can FORCE_EXPORT_DEPS: khong export YOLO)
#   2. bash docker/fetch_peoplenet_transformer.sh   (tu goc repo)
#   3. bash docker/vast_peoplenet_check.sh          (plugin + build engine b7 + throughput tho)
#   4. bash docker/vast_pnt.sh wildtrack pnt  configs/demo/streams_wildtrack_pnt.yaml  3
#      bash docker/vast_pnt.sh wildtrack pnt2 configs/demo/streams_wildtrack_pnt2.yaml 3
#   5. bash docker/vast_pnt.sh fps yolo configs/pipeline/streams_reid.yaml
#      bash docker/vast_pnt.sh fps pnt  configs/pipeline/streams_reid_pnt.yaml
#      bash docker/vast_pnt.sh fps pnt2 configs/pipeline/streams_reid_pnt2.yaml
#
# Buoc 4: ~3.5 phut/lan (400 khung 2 fps, sync=true). Fixture ra
# data/fixtures/ds_wildtrack_7cam_<tag>_r<n>.jsonl, cung quy uoc ten voi phien 25 nen
# cham duoc bang dung cac lenh cu. Doi chung YOLO11s 640: cham LAI 3 fixture R640 cua
# phien 25 bang code hien tai (khac may giua hai phien 22/25 da do: trong nhieu).
# Buoc 5: moi cau hinh chay 2 lan, lay lan 2 (lan 1 co the con build engine ReID/YOLO).
#
# Chay lau thi boc bang nohup ... < /dev/null &: ssh rot giua chung se giet tien trinh.
set -euo pipefail
REPO=/workspace/mct-repo
cd "$REPO"   # nvtracker phan giai duong dan theo cwd (CLAUDE.md §11)
mkdir -p data/fixtures logs

step="${1:?buoc: wildtrack <tag> <streams> <so-lan> | fps <tag> <streams>}"
tag="${2:?tag}"
streams="${3:?streams yaml}"
[ -f "$streams" ] || { echo "FATAL: khong co $streams"; exit 1; }

# DeepStream-Yolo ghi engine vao <cwd>, khong vao cho config khai (CLAUDE.md §11).
# PeopleNet Transformer khong dinh: nvinfer tu ghi dung ten.
cat_engine_yolo() {
  for b in 4 7; do
    if [ -f "model_b${b}_gpu0_fp16.engine" ] && [ ! -f "models/detector/yolo11s.onnx_b${b}_gpu0_fp16.engine" ]; then
      mv "model_b${b}_gpu0_fp16.engine" "models/detector/yolo11s.onnx_b${b}_gpu0_fp16.engine"
      echo "engine YOLO b${b} -> models/detector/"
    fi
  done
}

# Ghi nhiet do / xung GPU suot lan do (moi 5 s). Phien 30: may T4 dau tien bi SW Thermal
# Slowdown (84 C, SM 300/1590 MHz) -> throughput PeopleNet v1 con 1/4 va pipeline khong
# theo kip 14 anh/s cua WildTrack. So do ma khong kem bang chung nay thi khong tin duoc.
gpu_log_start() {
  nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks_throttle_reasons.active,utilization.gpu \
    --format=csv,noheader,nounits -l 5 > "$1" 2>/dev/null &
  GPU_LOG_PID=$!
}
gpu_log_stop() {
  kill "$GPU_LOG_PID" 2>/dev/null || true
  python3 - "$1" <<'PY'
import sys
rows = [r.split(", ") for r in open(sys.argv[1]) if r.count(",") >= 4]
busy = [r for r in rows if int(r[4]) >= 50]
if not busy:
    print("gpu: khong co mau nao GPU >= 50% util"); sys.exit()
temp = max(int(r[1]) for r in busy); clk = min(int(r[2]) for r in busy)
thr = sorted({r[3] for r in busy} - {"0x0000000000000000"})
print(f"gpu ({len(busy)} mau ban): nhiet max {temp} C, xung SM min {clk} MHz, throttle {thr or 'khong'}")
PY
}

# Bang chung bac bo duoc cho 3 bay cua phien 30: probe giu dung lop, engine duoc NAP LAI
# (khong build lai), va co detection that (fixture/khung khong rong).
bang_chung() {
  local log="$1"
  grep -h -o "probe giữ lớp [0-9]* của detector làm person" "$log" | head -1 || true
  grep -h -E "deserialized|Trying to create engine|serialize cuda engine" "$log" | sed 's/^.*INFO: //' | head -3 || true
  grep -h -E "throughput:|  cam0[0-9]: " "$log" | sed 's/^.*INFO *[^ ]* *//' || true
}

case "$step" in
wildtrack)
  nrun="${4:?so lan chay}"
  redis-cli ping >/dev/null 2>&1 || { redis-server --daemonize yes --save '' --appendonly no; sleep 1; }
  for n in $(seq 1 "$nrun"); do
    out=data/fixtures/ds_wildtrack_7cam_${tag}_r${n}.jsonl
    echo "=== ${tag} lan ${n}/${nrun} $(date -u +%H:%M:%S) ==="
    redis-cli FLUSHALL >/dev/null
    python3 -m tools.record_metadata --out "$out" > "logs/record_${tag}_r${n}.log" 2>&1 &
    rec=$!
    gpu_log_start "logs/gpu_${tag}_r${n}.csv"
    python3 -m ds_pipeline --config "$streams" --publish --stats > "logs/pipeline_${tag}_r${n}.log" 2>&1
    gpu_log_stop "logs/gpu_${tag}_r${n}.csv"
    sleep 5
    kill -TERM "$rec"; wait "$rec" || true
    cat_engine_yolo
    bang_chung "logs/pipeline_${tag}_r${n}.log"
    python3 - "$out" <<'EOF'
import json, sys
n_msg = n_det = 0
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        n_msg += 1
        n_det += len(json.loads(line).get("detections", []))
print(f"fixture: {sys.argv[1]}  {n_msg} message (ky vong 2800), {n_det} detection"
      f" ({n_det / max(n_msg, 1):.1f}/khung)")
if n_det == 0:
    sys.exit("FATAL: fixture KHONG co detection — kiem pgie.person_class_id")
EOF
  done
  echo "=== ${tag} XONG $(date -u +%H:%M:%S) ==="
  ;;

fps)
  for i in 1 2; do
    log=logs/fps_${tag}_${i}.log
    echo "=== FPS ${tag} lan ${i}/2 $(date -u +%H:%M:%S) ==="
    gpu_log_start "logs/gpu_fps_${tag}_${i}.csv"
    timeout 1200 python3 -m ds_pipeline --config "$streams" --stats > "$log" 2>&1 || true
    gpu_log_stop "logs/gpu_fps_${tag}_${i}.csv"
    cat_engine_yolo
    bang_chung "$log"
  done
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
  ;;

*)
  echo "buoc khong hop le: $step"; exit 1
  ;;
esac
