#!/bin/bash
# Du lieu TU THU (M6) tren instance vast.ai — quy trinh day du o docs/m6/README.md.
# Chay TREN instance, tu goc repo, theo thu tu:
#
#   0. bash docker/vast_latency.sh nvdec        LAM DAU TIEN (CLAUDE.md §11, 30 giay)
#   1. bash docker/vast_bootstrap.sh            (khong can FORCE_EXPORT_DEPS)
#   2. bash docker/vast_lab.sh ffmpeg           sua ffmpeg cua image DS 7.1 (chi khi can buoc 3)
#   3. bash docker/vast_lab.sh sync s1          dong bo + CFR video tho data/lab/raw/s1/cam0N.*
#        (bo qua neu da dong bo tren may khac va da day len data/lab/s1/cam0N.mp4)
#   4. bash docker/vast_lab.sh engine s1        build engine TensorRT (b4 YOLO + ReID), ~3-5 phut
#   5. bash docker/vast_lab.sh run s1 3         3 lan chay pipeline: fixture + log do tre dong ho that
#   6. bash docker/vast_lab.sh fps s1           FPS 4 luong khong sync (hai lan, lay lan 2)
#   7. bash docker/vast_lab.sh pack s1          gom ket qua -> data/lab_s1_results.tar
#   Tren may dev: scp vast-gpu:/workspace/mct-repo/data/lab_s1_results.tar . && tar xf ...
#   roi:  python -m eval.run_lab_eval ... (docs/m6/README.md, buoc "Cham")
#   NHO HUY MAY: vastai destroy instance <id> -y  (tinh tien theo gio).
#
# Buoc 5 moi lan = thoi luong video (sync=true) + ~30 s. Moi lan chay cho ra:
#   data/fixtures/lab_<buoi>_r<n>.jsonl        fixture (hop + id NvDCF + embedding)
#   data/lab/<buoi>/latency_r<n>.jsonl          moc t0..t4 theo dong ho THAT (dinh nghia phien 32)
#   data/lab/<buoi>/engine_r<n>.db              SQLite cua engine chay online cung luc
#   logs/{pipeline,engine,record,gpu}_lab_<buoi>_r<n>.*
# Engine o day dung configs/lab/lab.mct.yaml + topology + homography cua lab: day la lan dau
# (B) "thoi gian toi Global ID" duoc do bang dong ho that tren du lieu 25 fps (worklog phien
# 32, "vuong mac"). So do chinh xac van cham lai tren may dev tu fixture (engine tat dinh).
#
# Chay lau thi boc bang nohup ... < /dev/null &: ssh rot giua chung se giet tien trinh.
set -euo pipefail
REPO=/workspace/mct-repo
cd "$REPO"   # nvtracker phan giai duong dan theo cwd (CLAUDE.md §11)
mkdir -p data/fixtures logs

step="${1:?buoc: ffmpeg | sync <buoi> | engine <buoi> | run <buoi> <so-lan> | fps <buoi> | pack <buoi>}"
STREAMS=configs/lab/streams_lab.yaml
ENGINE_CFG=configs/lab/lab.mct.yaml
TOPO=configs/lab/topology.yaml
HDIR=configs/lab/homography
IDLE_LIMIT=15

# Giong vast_pnt.sh: ghi nhiet do / xung GPU suot lan do (phien 30: T4 ha xung 300 MHz).
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

# DeepStream-Yolo ghi engine vao <cwd>, khong vao cho config khai (CLAUDE.md §11).
cat_engine_yolo() {
  if [ -f model_b4_gpu0_fp16.engine ] && [ ! -f models/detector/yolo11s.onnx_b4_gpu0_fp16.engine ]; then
    mv model_b4_gpu0_fp16.engine models/detector/yolo11s.onnx_b4_gpu0_fp16.engine
    echo "engine YOLO b4 -> models/detector/"
  fi
}

need_session() {
  session="${1:?ten buoi quay, vd s1}"
  export LAB_SESSION="$session"
  for c in cam01 cam02 cam03 cam04; do
    [ -f "data/lab/$session/$c.mp4" ] || { echo "FATAL: thieu data/lab/$session/$c.mp4 (chay buoc sync?)"; exit 1; }
  done
}

case "$step" in
ffmpeg)
  # Image DS 7.1 co /usr/bin/ffmpeg nhung thieu thu vien codec (CLAUDE.md §11).
  apt-get install -y libflac8 libmp3lame0 libxvidcore4 >/dev/null
  ffmpeg -hide_banner -encoders 2>/dev/null | grep -c libx264 >/dev/null \
    || { echo "FATAL: ffmpeg van khong co libx264"; exit 1; }
  echo "ffmpeg OK (libx264)"
  ;;

sync)
  session="${2:?ten buoi quay}"
  raw="data/lab/raw/$session"
  args=()
  for c in cam01 cam02 cam03 cam04; do
    f=$(ls "$raw/$c".* 2>/dev/null | head -1 || true)
    [ -n "$f" ] || { echo "FATAL: khong co $raw/$c.*"; exit 1; }
    args+=(--video "$c=$f")
  done
  python3 -m tools.sync_recordings "${args[@]}" --out-dir "data/lab/$session" --fps 25 --execute
  ;;

engine)
  need_session "${2:-}"
  # Build engine trong mot lan RIENG khong sync: lan do do tre ma phai build (~3-5 phut)
  # thi t1-t0 cua cac khung dau bi doc them thoi gian build (vast_latency.sh).
  sed 's/^  sync: true/  sync: false/' "$STREAMS" > /tmp/streams_lab_nosync.yaml
  timeout 1200 python3 -m ds_pipeline --config /tmp/streams_lab_nosync.yaml --stats 2>&1 | tail -15
  cat_engine_yolo
  ls -la models/detector/*.engine 2>/dev/null || true
  ;;

run)
  need_session "${2:-}"
  nrun="${3:?so lan chay (3, phien 22: nhieu giua hai lan chay ~0.3-0.9 HOTA)}"
  command -v redis-server >/dev/null || { echo "FATAL: chua co redis-server"; exit 1; }
  redis-cli ping >/dev/null 2>&1 || { redis-server --daemonize yes --save '' --appendonly no; sleep 1; }
  python3 -m tools.check_lab_setup --session "$session" || {
    echo "FATAL: check_lab_setup bao FAIL — sua cau hinh tren may dev truoc khi dot tien GPU"; exit 1; }
  for n in $(seq 1 "$nrun"); do
    tag="lab_${session}_r${n}"
    out="data/fixtures/${tag}.jsonl"
    lat="data/lab/$session/latency_r${n}.jsonl"
    db="data/lab/$session/engine_r${n}.db"
    echo "=== ${tag} $(date -u +%H:%M:%S) ==="
    redis-cli FLUSHALL >/dev/null
    rm -f "$lat" "$db" "$db-wal" "$db-shm"
    python3 -m tools.record_metadata --out "$out" > "logs/record_${tag}.log" 2>&1 &
    rec=$!
    MCT_DB_PATH="$db" nohup python3 -m mct --config "$ENGINE_CFG" --topology "$TOPO" \
        --homography-dir "$HDIR" --db "$db" --publish --latency-log "$lat" \
        --idle-limit "$IDLE_LIMIT" > "logs/engine_${tag}.log" 2>&1 &
    eng=$!
    sleep 3   # engine kip tao consumer group truoc message dau tien
    gpu_log_start "logs/gpu_${tag}.csv"
    python3 -m ds_pipeline --config "$STREAMS" --publish --stats > "logs/pipeline_${tag}.log" 2>&1 || true
    gpu_log_stop "logs/gpu_${tag}.csv"
    for _ in $(seq $((IDLE_LIMIT + 30))); do kill -0 "$eng" 2>/dev/null || break; sleep 1; done
    kill -0 "$eng" 2>/dev/null && { kill -TERM "$eng"; sleep 5; }
    kill -TERM "$rec"; wait "$rec" || true
    cat_engine_yolo
    grep -h -E "throughput:|  cam0[0-9]: " "logs/pipeline_${tag}.log" | tail -6 || true
    python3 - "$out" <<'EOF'
import json, sys
n_msg = n_det = 0
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        n_msg += 1
        n_det += len(json.loads(line).get("detections", []))
print(f"fixture: {sys.argv[1]}  {n_msg} message, {n_det} detection ({n_det / max(n_msg, 1):.1f}/khung)")
if n_det == 0:
    sys.exit("FATAL: fixture KHONG co detection")
EOF
    python3 -m tools.latency_report --log "$lat" --top 3 2>&1 | tail -25 || true
  done
  echo "=== run XONG $(date -u +%H:%M:%S) ==="
  ;;

fps)
  need_session "${2:-}"
  sed 's/^  sync: true/  sync: false/' "$STREAMS" > /tmp/streams_lab_nosync.yaml
  for i in 1 2; do
    log="logs/fps_lab_${session}_${i}.log"
    echo "=== FPS lan ${i}/2 $(date -u +%H:%M:%S) ==="
    gpu_log_start "logs/gpu_fps_lab_${session}_${i}.csv"
    timeout 1800 python3 -m ds_pipeline --config /tmp/streams_lab_nosync.yaml --stats > "$log" 2>&1 || true
    gpu_log_stop "logs/gpu_fps_lab_${session}_${i}.csv"
    grep -h -E "throughput:|  cam0[0-9]: " "$log" | tail -6 || true
  done
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
  ;;

pack)
  session="${2:?ten buoi quay}"
  tar cf "data/lab_${session}_results.tar" data/fixtures/lab_"${session}"_r*.jsonl \
      data/lab/"$session"/latency_r*.jsonl data/lab/"$session"/engine_r*.db \
      data/lab/"$session"/sync.json logs/*lab_"${session}"* 2>/dev/null
  ls -la "data/lab_${session}_results.tar"
  ;;

*)
  echo "buoc khong hop le: $step"; exit 2
  ;;
esac
