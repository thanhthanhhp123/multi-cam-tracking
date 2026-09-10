#!/bin/bash
# WildTrack 7 camera tren instance vast.ai: tai + dong video, roi chay pipeline ghi fixture
# voi mot gia tri pre-cluster-threshold. Chay TREN instance, SAU docker/vast_bootstrap.sh.
# Dung lan dau o phien 22 (docs/worklog/2026-09-10-22-*) de sinh lai fixture khi ut-hpc mat.
#
#   scp docker/vast_wildtrack.sh vast-gpu:/workspace/
#   ssh vast-gpu 'bash /workspace/vast_wildtrack.sh nvdec'      # LAM DAU TIEN (CLAUDE.md §11)
#   ssh vast-gpu 'bash /workspace/vast_wildtrack.sh data'       # ~10 phut
#   ssh vast-gpu 'bash /workspace/vast_wildtrack.sh run 0.25'   # ~3.5 phut (+5 phut build engine lan dau)
#
# Da co video tren may dev (data/wildtrack_video/)? Bo buoc `data`, day len bang:
#   tar cf - data/wildtrack_video | ssh vast-gpu 'tar xf - -C /workspace/mct-repo'
#
# Chay lau thi boc bang nohup ... < /dev/null &: ssh rot giua chung se giet tien trinh.
# `run` ghi de fixture cung nguong — muon lap thi doi ten fixture/log sau moi lan
# (nhieu giua cac lan chay ~0.3-0.9 HOTA, xem worklog phien 22 QD 1).
set -euo pipefail
REPO=/workspace/mct-repo
ZIP_URL="https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip"
ZIP=/workspace/Wildtrack_dataset_full.zip

step="${1:?buoc: nvdec | data | run <nguong>}"
case "$step" in
nvdec)
  SAMPLE=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  timeout 60 gst-launch-1.0 -e nvurisrcbin uri=file://$SAMPLE ! fakesink sync=false 2>&1 \
    | grep -E "PREROLLED|PLAYING|EOS" || { echo "FATAL: NVDEC hong — huy may"; exit 1; }
  ;;
data)
  cd "$REPO"
  # Image DeepStream 7.1 co /usr/bin/ffmpeg nhung thieu thu vien codec (go vi ban quyen).
  apt-get install -y -qq libflac8 libmp3lame0 libxvidcore4 >/dev/null 2>&1 \
    || { apt-get update -qq && apt-get install -y -qq ffmpeg libflac8 libmp3lame0 libxvidcore4 >/dev/null; }
  # grep -c chu khong phai grep -q: -q thoat som -> ffmpeg nhan SIGPIPE -> pipefail bao loi gia.
  ffmpeg -hide_banner -encoders 2>/dev/null | grep -c libx264 >/dev/null || { echo "FATAL: ffmpeg thieu libx264"; exit 1; }
  if [ ! -f "$ZIP" ]; then
    wget -q -O "$ZIP.part" "$ZIP_URL"
    mv "$ZIP.part" "$ZIP"
  fi
  ls -la "$ZIP"
  # Chu thich (annotations_positions/) phai co san trong data/wildtrack — day tu may dev.
  python3 -m tools.unzip_wildtrack --zip "$ZIP" --dest data/wildtrack/Image_subsets
  python3 -m tools.wildtrack_to_video --wildtrack-dir data/wildtrack --out-dir data/wildtrack_video
  ls -la data/wildtrack_video
  ;;
run)
  cd "$REPO"   # nvtracker phan giai duong dan ONNX theo THU MUC LAM VIEC (CLAUDE.md §11)
  thr="${2:?nguong, vd 0.25}"
  tag="c${thr/./}"
  infer=configs/pipeline/config_infer_yolo11_b7_${tag}.txt
  streams=configs/demo/streams_wildtrack_${tag}.yaml
  # Chi dong [class-attrs-0] co gia tri 0.25; dong [class-attrs-all] la 1.0 nen khong bi dung.
  sed "s/^pre-cluster-threshold=0.25\$/pre-cluster-threshold=${thr}/" \
    configs/pipeline/config_infer_yolo11_b7.txt > "$infer"
  [ "$(grep -c "^pre-cluster-threshold=${thr}\$" "$infer")" = 1 ] || { echo "FATAL: sed khong khop"; exit 1; }
  sed "s#config_infer_yolo11_b7.txt#config_infer_yolo11_b7_${tag}.txt#" \
    configs/demo/streams_wildtrack.yaml > "$streams"
  grep -q "_${tag}.txt" "$streams" || { echo "FATAL: streams khong tro toi config moi"; exit 1; }

  if [ "$thr" = "0.25" ]; then out=data/fixtures/ds_wildtrack_7cam.jsonl
  else out=data/fixtures/ds_wildtrack_7cam_${tag}.jsonl; fi
  mkdir -p data/fixtures logs
  redis-cli ping >/dev/null 2>&1 || redis-server --daemonize yes
  sleep 1
  redis-cli FLUSHALL >/dev/null   # nhom 'recorder' doc tu dau stream -> phai xoa lan chay truoc

  python3 -m tools.record_metadata --out "$out" > "logs/record_${tag}.log" 2>&1 &
  rec=$!
  python3 -m ds_pipeline --config "$streams" --publish 2>&1 | tee "logs/pipeline_${tag}.log"
  sleep 5
  kill -TERM "$rec"; wait "$rec" || true

  # DeepStream-Yolo ghi engine vao <cwd>, khong vao cho config khai (CLAUDE.md §11).
  eng=models/detector/yolo11s.onnx_b7_gpu0_fp16.engine
  [ -f "$eng" ] || { [ -f model_b7_gpu0_fp16.engine ] && cp model_b7_gpu0_fp16.engine "$eng"; }
  echo "fixture: $out  $(wc -l < "$out") message (ky vong 2800)"
  tail -3 "logs/record_${tag}.log"
  ;;
*)
  echo "buoc la: $step"; exit 2 ;;
esac
