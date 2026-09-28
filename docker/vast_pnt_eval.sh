#!/bin/bash
# Cham diem fixture phien 30 NGAY TREN instance (nguoi dung chon de khong tai may dev
# dang yeu). Cung duong cham voi phien 24-26, chi khac noi chay:
#   box    = eval.compare_oracle_tracker, kich ban A (hop anh IoU 0.5, toan khung, 400 khung)
#   ground = eval.eval_ground_plane (diem mat dat, trong vung, T = 1 m, NMS 0 va 0.5 m;
#            in ca 400 khung lan 40 khung test)
# Doi chung YOLO11s 640 = 3 fixture R640 cua phien 25, cham LAI bang code hien tai de moi
# so trong bang cung mot phien ban src/mct + eval.
#
#   bash docker/vast_pnt_eval.sh setup     # TrackEval (commit ghim, nhu may dev) + venv numpy 1.23.5
#   bash docker/vast_pnt_eval.sh gt        # gan ground-truth cho fixture moi (tools.ds_wildtrack_gt)
#   bash docker/vast_pnt_eval.sh box       # -> data/s30/compare_oracle_tracker.json
#   bash docker/vast_pnt_eval.sh ground    # -> data/s30/ground_plane.json
set -euo pipefail
REPO=/workspace/mct-repo
cd "$REPO"
TE=/workspace/TrackEval
TE_REF=12c8791b303e0a0b50f753af204249e622d0281a   # = ~/TrackEval tren may dev
VENV=/workspace/venv-eval                          # TrackEval ghim numpy 1.23.5
F=data/fixtures
TAGS=(r640n:R640 pnt:PNT pnt2:PNT2)

runs() {  # runs box|ground -> cac cap --run cho moi fixture co san
  local mode="$1" pair tag label n f
  for pair in "${TAGS[@]}"; do
    tag=${pair%%:*}; label=${pair##*:}
    for n in 1 2 3; do
      f=$F/ds_wildtrack_7cam_${tag}_r${n}.jsonl
      [ -f "$f" ] || continue
      if [ "$mode" = box ]; then
        echo "--run ${label}:r${n} $f ${f%.jsonl}.gt.json"
      else
        echo "--run ${label}:r${n} $f data/s30/${label}_r${n}/mct.db"
      fi
    done
  done
}

step="${1:?buoc: setup | gt | box | ground}"
case "$step" in
setup)
  [ -d "$TE/.git" ] || git clone -q https://github.com/JonathonLuiten/TrackEval "$TE"
  git -C "$TE" checkout -q "$TE_REF"
  # uv, khong dung `python3 -m venv`: image DeepStream thieu ensurepip (python3.10-venv),
  # venv tao ra khong co pip (gap o phien 30).
  command -v uv >/dev/null || pip install -q uv
  [ -x "$VENV/bin/python" ] || uv venv -q --clear --python "$(command -v python3)" "$VENV"
  uv pip install -q --python "$VENV/bin/python" \
    numpy==1.23.5 scipy==1.10.1 msgpack PyYAML python-dotenv redis
  "$VENV/bin/python" -c "import numpy, scipy; print('venv-eval numpy', numpy.__version__, 'scipy', scipy.__version__)"
  ;;
gt)
  for f in $F/ds_wildtrack_7cam_pnt_r*.jsonl $F/ds_wildtrack_7cam_pnt2_r*.jsonl; do
    [ -f "${f%.jsonl}.gt.json" ] && { echo "da co ${f%.jsonl}.gt.json"; continue; }
    PYTHONPATH=src python3 -m tools.ds_wildtrack_gt --fixture "$f" --wildtrack-dir data/wildtrack \
      --report "${f%.jsonl}.gt-report.json" 2>&1 | tail -3
  done
  ;;
box)
  # shellcheck disable=SC2046  # tach tu co chu dich: moi --run la 3 doi so
  PYTHONPATH=src python3 -m eval.compare_oracle_tracker \
    --config configs/demo/wildtrack_ds.mct.yaml \
    --topology configs/demo/wildtrack.topology.yaml \
    --homography-dir configs/cameras/homography/wildtrack \
    --gt-fixture $F/wildtrack_7cam.jsonl --gt-fixture-table $F/wildtrack_7cam.gt.json \
    --engine-python python3 --eval-python "$VENV/bin/python" --trackeval-path "$TE" \
    --work-dir data/s30 $(runs box)
  ;;
ground)
  # shellcheck disable=SC2046
  PYTHONPATH="src:." "$VENV/bin/python" -m eval.eval_ground_plane \
    --trackeval-path "$TE" --wildtrack-dir data/wildtrack \
    --homography-dir configs/cameras/homography/wildtrack \
    --threshold-m 1.0 --area in --nms-m 0 0.5 \
    $(runs ground) --json data/s30/ground_plane.json
  ;;
*)
  echo "buoc khong hop le: $step"; exit 1
  ;;
esac
