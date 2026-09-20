#!/bin/bash
# Quet DO PHAN GIAI SUY LUAN cua detector (YOLO11s) tren WildTrack 7 cam — phien 25
# (docs/worklog/2026-09-20-25-*). Chay TREN instance vast.ai, SAU docker/vast_bootstrap.sh.
# Doi dung MOT bien: kich thuoc dau vao vuong cua ONNX; moi thu khac giong
# configs/pipeline/config_infer_yolo11_b7.txt (pre-cluster-threshold 0.25, nms 0.45, topk 300).
#
# Chuan bi (mot lan, tren instance):
#   1. Day len /workspace/mct-repo: `git archive HEAD`, models/, data/wildtrack/annotations_positions
#      va data/wildtrack_video (xem docker/vast_wildtrack.sh).
#   2. numpy PHAI la 1.x: neu da chay vast_bootstrap.sh voi FORCE_EXPORT_DEPS=1 (ultralytics)
#      thi pip da nang numpy len 2.x va nvtracker SEGFAULT (exit 139) luc build engine ReID.
#      Chay `pip install numpy==1.26.4` sau khi export xong (CLAUDE.md §11).
#   3. Export ONNX cho tung kich thuoc — export_yolo11.py yeu cau yolo11s.pt CO SAN o cwd
#      (khong tu tai), nen tai truoc bang ultralytics:
#        mkdir -p /workspace/exp && cd /workspace/exp && python3 -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"
#        for s in 640 960 1280; do mkdir -p s$s && cd s$s && cp ../yolo11s.pt . #          && python3 /workspace/DeepStream-Yolo/utils/export_yolo11.py -w yolo11s.pt -s $s --dynamic; cd ..; done
#
# Chay:   bash docker/vast_detector_res.sh <tag> <kich-thuoc-export> <so-lan>     vd: r960 960 3
# Ket qua: data/fixtures/ds_wildtrack_7cam_<tag>_r<n>.jsonl (ky vong 2800 message moi cai).
# Lan chay dau cua moi tag build engine detector (640 ~4 phut, 960 ~9, 1280 ~15 tren T4) —
# im lang KHONG co nghia la treo, dung kill (CLAUDE.md §11). Log in dong "nvinfer input: ...3xSxS"
# de bac bo duoc viec config bi bo qua im lang.
set -euo pipefail
REPO=/workspace/mct-repo
tag="${1:?tag}"; size="${2:?size}"; nrun="${3:?so lan chay}"
cd "$REPO"   # nvtracker phan giai duong dan theo cwd (CLAUDE.md §11)

src=/workspace/exp/s${size}
det=models/detector/${tag}
mkdir -p "$det" data/fixtures logs
cp "$src/yolo11s.onnx" "$src/yolo11s.onnx.data" models/detector/labels.txt "$det/"

infer=configs/pipeline/config_infer_yolo11_b7_${tag}.txt
streams=configs/demo/streams_wildtrack_${tag}.yaml
sed -e "s#^onnx-file=../../models/detector/yolo11s.onnx#onnx-file=../../${det}/yolo11s.onnx#" \
    -e "s#^model-engine-file=../../models/detector/yolo11s.onnx_b7_gpu0_fp16.engine#model-engine-file=../../${det}/yolo11s.onnx_b7_gpu0_fp16.engine#" \
    -e "s#^labelfile-path=../../models/detector/labels.txt#labelfile-path=../../${det}/labels.txt#" \
    configs/pipeline/config_infer_yolo11_b7.txt > "$infer"
[ "$(grep -c "^onnx-file=../../${det}/yolo11s.onnx\$" "$infer")" = 1 ] || { echo "FATAL: sed onnx-file khong khop"; exit 1; }
[ "$(grep -c "^model-engine-file=../../${det}/" "$infer")" = 1 ] || { echo "FATAL: sed engine khong khop"; exit 1; }
[ "$(grep -c "^labelfile-path=../../${det}/" "$infer")" = 1 ] || { echo "FATAL: sed labels khong khop"; exit 1; }
grep -q "^pre-cluster-threshold=0.25\$" "$infer" || { echo "FATAL: nguong 0.25 mat"; exit 1; }
sed "s#config_infer_yolo11_b7.txt#config_infer_yolo11_b7_${tag}.txt#" \
    configs/demo/streams_wildtrack.yaml > "$streams"
grep -q "_${tag}.txt" "$streams" || { echo "FATAL: streams khong tro toi config moi"; exit 1; }

redis-cli ping >/dev/null 2>&1 || redis-server --daemonize yes
sleep 1

for n in $(seq 1 "$nrun"); do
  out=data/fixtures/ds_wildtrack_7cam_${tag}_r${n}.jsonl
  echo "=== ${tag} lan ${n}/${nrun} $(date -u +%H:%M:%S) ==="
  redis-cli FLUSHALL >/dev/null
  python3 -m tools.record_metadata --out "$out" > "logs/record_${tag}_r${n}.log" 2>&1 &
  rec=$!
  python3 -m ds_pipeline --config "$streams" --publish > "logs/pipeline_${tag}_r${n}.log" 2>&1
  sleep 5
  kill -TERM "$rec"; wait "$rec" || true

  # DeepStream-Yolo ghi engine vao <cwd>, khong vao cho config khai (CLAUDE.md §11).
  eng="${det}/yolo11s.onnx_b7_gpu0_fp16.engine"
  if [ ! -f "$eng" ] && [ -f model_b7_gpu0_fp16.engine ]; then
    mv model_b7_gpu0_fp16.engine "$eng"
    echo "engine build lan dau -> $eng ($(du -h "$eng" | cut -f1))"
  fi
  # Bang chung bac bo duoc: engine that su co dau vao dung kich thuoc (khong tin config).
  echo "nvinfer input: $(grep -h -o -E 'INPUT.*input +[0-9]+x[0-9]+x[0-9]+' "logs/pipeline_${tag}_r${n}.log" | head -1)"
  echo "fixture: $out  $(wc -l < "$out") message (ky vong 2800)"
done
echo "=== ${tag} XONG $(date -u +%H:%M:%S) ==="
