#!/usr/bin/env bash
# Helper for driving the ut-hpc SLURM cluster from the dev machine.
# All heavy/GPU work goes through sbatch; anything touching the network runs on the head
# node, because compute nodes have no internet access. See ../reference/cluster.md.
set -euo pipefail

HOST="${UT_HPC_HOST:-ut-hpc}"
REMOTE_ROOT="${UT_HPC_ROOT:-mct}"          # relative to $HOME on the cluster
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST")
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "error: $*" >&2; exit 1; }

remote() { "${SSH[@]}" "$@"; }

cmd_status() {
  remote "bash -s" <<'RS'
set -u
echo "=== ket noi ==="
hostname; uptime | sed 's/^/  /'
echo "=== job cua toi ==="
squeue -u "$USER" -o "%.10i %.10P %.20j %.2t %.10M %.6D %R" || true
echo "=== GPU ranh tren main-gpu ==="
sinfo -p main-gpu -o "%.14N %.6t %.30G" -h | head -30
echo "=== dung luong home ==="
df -h "$HOME" | tail -1
echo "=== thu muc du an ==="
ls -la "$HOME/mct" 2>/dev/null || echo "  (chua co ~/mct — chay: hpc.sh setup)"
RS
}

cmd_setup() {
  echo ">> Tao cay thu muc + conda env 'mct' tren HEAD NODE (co internet). Mat ~10 phut."
  remote "bash -s" <<'RS'
set -eu
mkdir -p "$HOME/mct"/{data,jobs,runs,models,logs,.torch}
ENV_DIR="$HOME/mct/env"
if [ -x "$ENV_DIR/bin/python" ]; then
  echo "env da co: $ENV_DIR"
  "$ENV_DIR/bin/python" --version
  exit 0
fi
CONDA=""
for c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$(command -v conda || true)"; do
  [ -n "$c" ] && [ -x "$c" ] && { CONDA="$c"; break; }
done
[ -n "$CONDA" ] || { echo "khong tim thay conda"; exit 1; }
echo ">> conda: $CONDA"
"$CONDA" create -y -p "$ENV_DIR" python=3.10
"$ENV_DIR/bin/pip" install -q -U pip
# CUDA wheels: driver 595 chay duoc moi ban cu12x. Giu cu124 cho on dinh voi ultralytics.
"$ENV_DIR/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
"$ENV_DIR/bin/pip" install -q ultralytics onnx onnxruntime onnxslim pyyaml
"$ENV_DIR/bin/python" -c "import torch, ultralytics; print('torch', torch.__version__); print('ultralytics', ultralytics.__version__)"
echo ">> xong: $ENV_DIR"
df -h "$HOME" | tail -1
RS
}

cmd_push() {
  local src="$SKILL_DIR/templates"
  [ -d "$src" ] || die "khong thay $src"
  echo ">> day templates/ -> $HOST:~/$REMOTE_ROOT/jobs/"
  remote "mkdir -p \$HOME/$REMOTE_ROOT/{data,jobs,runs,models,logs,.torch}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -av -e "ssh -o BatchMode=yes" "$src"/ "$HOST:$REMOTE_ROOT/jobs/"
  else
    scp -o BatchMode=yes -r "$src"/* "$HOST:$REMOTE_ROOT/jobs/"
  fi
}

cmd_submit() {
  local job="${1:-}"; shift || true
  [ -n "$job" ] || die "dung: hpc.sh submit <ten-file.sbatch> [tham so...]"
  local out
  out=$(remote "cd \$HOME/$REMOTE_ROOT/jobs && sbatch $job $*") || die "sbatch that bai"
  echo "$out"
  echo "$out" | grep -oE '[0-9]+$' | tail -1 > /dev/null || true
}

cmd_watch() {
  local jid="${1:-}"
  [ -n "$jid" ] || die "dung: hpc.sh watch <jobid>"
  echo ">> theo doi job $jid (Ctrl-C de thoat; job VAN chay tiep)"
  remote "bash -s" <<RS
set -u
for i in \$(seq 1 720); do
  line=\$(squeue -j $jid -h -o "%T %M %R" 2>/dev/null || true)
  [ -z "\$line" ] && break
  echo "[\$i] \$line"
  sleep 10
done
echo "=== sacct ==="
sacct -j $jid --format=JobID,JobName%20,Partition,NodeList,State,Elapsed,MaxRSS 2>/dev/null | head -5
echo "=== log ==="
tail -60 "\$HOME/$REMOTE_ROOT/logs/slurm-$jid.out" 2>/dev/null || echo "(chua co log)"
RS
}

cmd_logs() {
  local jid="${1:-}"
  [ -n "$jid" ] || die "dung: hpc.sh logs <jobid>"
  remote "cat \$HOME/$REMOTE_ROOT/logs/slurm-$jid.out"
}

cmd_cancel() {
  local jid="${1:-}"
  if [ -n "$jid" ]; then
    remote "scancel $jid && echo 'da huy $jid'"
  else
    echo ">> huy TAT CA job cua \$USER"
    remote 'scancel -u $USER; sleep 2; squeue -u $USER'
  fi
}

cmd_fetch() {
  local rpath="${1:-}" lpath="${2:-}"
  [ -n "$rpath" ] && [ -n "$lpath" ] || die "dung: hpc.sh fetch <duong-dan-tren-cum> <duong-dan-local>"
  mkdir -p "$(dirname "$lpath")"
  scp -o BatchMode=yes "$HOST:$rpath" "$lpath"
  echo ">> da tai ve $lpath"
  ls -lh "$lpath"
}

cmd_shell() { exec ssh "$HOST"; }

usage() {
  cat <<'U'
hpc.sh — dieu khien cum ut-hpc tu may dev

  status                      ket noi, GPU ranh, job dang chay, dung luong home
  setup                       tao ~/mct + conda env "mct" (chay tren HEAD NODE, co mang)
  push                        day templates/*.sbatch|*.py len ~/mct/jobs/
  submit <file.sbatch> [args] sbatch, in JOBID
  watch  <jobid>              poll squeue toi khi xong roi in log
  logs   <jobid>              in log day du
  cancel [jobid]              huy 1 job, hoac tat ca neu khong truyen jobid
  fetch  <remote> <local>     scp trong so ve may dev (vd: models/detector/*.onnx)
  shell                       ssh vao head node

Bien moi truong: UT_HPC_HOST (mac dinh "ut-hpc"), UT_HPC_ROOT (mac dinh "mct").
U
}

case "${1:-}" in
  status) shift; cmd_status "$@" ;;
  setup)  shift; cmd_setup "$@" ;;
  push)   shift; cmd_push "$@" ;;
  submit) shift; cmd_submit "$@" ;;
  watch)  shift; cmd_watch "$@" ;;
  logs)   shift; cmd_logs "$@" ;;
  cancel) shift; cmd_cancel "$@" ;;
  fetch)  shift; cmd_fetch "$@" ;;
  shell)  shift; cmd_shell "$@" ;;
  ""|-h|--help|help) usage ;;
  *) die "lenh khong biet: $1 (xem: hpc.sh help)" ;;
esac
