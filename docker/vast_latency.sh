#!/bin/bash
# Do do tre theo tung chang (t0..t4) tren instance vast.ai. Chay TREN instance, SAU
# docker/vast_bootstrap.sh. Dung lan dau o phien 23 (docs/worklog/2026-09-18-23-*).
#
#   scp docker/vast_latency.sh vast-gpu:/workspace/
#   ssh vast-gpu 'bash /workspace/vast_latency.sh nvdec'    # LAM DAU TIEN (CLAUDE.md §11)
#   ssh vast-gpu 'bash /workspace/vast_latency.sh engine'   # build TensorRT engine truoc
#   ssh vast-gpu 'bash /workspace/vast_latency.sh run'      # ~1 phut + thoi luong video
#
# VI SAO CA HAI NUA CHAY TREN CUNG MOT MAY: `t2-t1b` va `t3a-t2` bac cau hai dong ho.
# Chay chung mot may thi moi doan nam trong MOT dong ho, khong lan do lech NTP vao so do.
# Ban phan tan (engine o may dev) do sau, khi da biet hinh dang cua duong co so nay.
set -euo pipefail

REPO=/workspace/mct-repo
CFG=configs/pipeline/streams_latency.yaml
OUT=data/latency.jsonl
IDLE_LIMIT=15   # engine tu thoat sau 15 lan cho 1s khong co message. Ngan lai vi vong gan
                # CUOI (final_flush) mang dung khoang cho nay vao window_wait — hien vat
                # cua phep do, khong phai hanh vi van hanh (docs/worklog/2026-09-18-23-*).

step="${1:?buoc: nvdec | engine | run}"

case "$step" in
nvdec)
  # 30 giay nay da tiet kiem ~30 phut tien thue mot lan roi (CLAUDE.md §11): card tieu
  # dung co the hong NVDEC du nvidia-smi/CUDA/TensorRT deu binh thuong.
  SAMPLE=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  timeout 60 gst-launch-1.0 -e nvurisrcbin uri=file://$SAMPLE ! fakesink sync=false 2>&1 \
    | grep -E "PREROLLED|PLAYING|EOS" || { echo "FATAL: NVDEC hong — huy may"; exit 1; }
  ;;

engine)
  # Build engine TensorRT trong mot lan chay RIENG, khong sync: lan chay do do tre dau
  # tien ma phai build engine (~3-5 phut) se co t1-t0 cua vai tram khung dau bi doc them
  # thoi gian build. Dung `--stats` de khong in tung khung.
  cd "$REPO"
  echo "=== build engine TensorRT (b4 YOLO + OSNet ReID), ~3-5 phut ==="
  timeout 900 python3 -m ds_pipeline --config configs/pipeline/streams_reid.yaml --stats 2>&1 \
    | tail -20
  ls -la /workspace/mct-repo/*.engine models/detector/*.engine 2>/dev/null || true
  ;;

run)
  cd "$REPO"
  command -v redis-server >/dev/null || { echo "FATAL: chua co redis-server"; exit 1; }
  redis-cli ping >/dev/null 2>&1 || { redis-server --daemonize yes --save '' --appendonly no; sleep 1; }
  redis-cli ping

  # Don sach: stream cu + consumer group cu lam engine doc lai message cua lan truoc.
  redis-cli DEL mct:frames mct:global >/dev/null
  rm -f "$OUT" engine.log pipeline.log data/mct.db data/mct.db-wal data/mct.db-shm
  mkdir -p data

  echo "=== engine (nen), ghi moc vao $OUT ==="
  MCT_DB_PATH=data/mct.db nohup python3 -m mct --config configs/mct.yaml --db data/mct.db \
      --publish --latency-log "$OUT" --idle-limit "$IDLE_LIMIT" > engine.log 2>&1 &
  engine_pid=$!
  sleep 3   # de engine kip tao consumer group truoc khi pipeline ban message dau tien

  echo "=== pipeline 4 luong CO ReID, sync=true ==="
  python3 -m ds_pipeline --config "$CFG" --publish --stats > pipeline.log 2>&1 || true
  tail -12 pipeline.log

  echo "=== cho engine vet not hang doi (toi da $((IDLE_LIMIT + 30))s) ==="
  # Khong dung `wait` tran: engine co the treo neu Redis chet giua chung.
  for _ in $(seq $((IDLE_LIMIT + 30))); do
    kill -0 "$engine_pid" 2>/dev/null || break
    sleep 1
  done
  kill -0 "$engine_pid" 2>/dev/null && { echo "engine van chay, gui SIGTERM"; kill -TERM "$engine_pid"; sleep 5; }
  tail -8 engine.log

  echo "=== ket qua ==="
  wc -l "$OUT"
  python3 -m tools.latency_report --log "$OUT" --top 5 2>&1 | tail -45
  ;;

*)
  echo "buoc khong hop le: $step"; exit 2
  ;;
esac
