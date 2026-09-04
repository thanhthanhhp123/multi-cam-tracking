#!/bin/bash
# Bật dashboard MTMCT trên ut-hpc để xem qua SSH port-forward.
#
#   bash ~/mct/dashboard-demo.sh start   # bật redis + dashboard, phát dữ liệu một lượt
#   bash ~/mct/dashboard-demo.sh feed    # phát thêm một lượt nữa (xem chấm chạy)
#   bash ~/mct/dashboard-demo.sh loop    # phát liên tục cho tới khi Ctrl-C / stop
#   bash ~/mct/dashboard-demo.sh status
#   bash ~/mct/dashboard-demo.sh stop
#
# Trên máy dev (Windows), mở một cửa sổ terminal riêng rồi chạy:
#   ssh -N -L 8000:localhost:8391 ut-hpc
# Sau đó vào http://localhost:8000

set -u
REPO=~/mct/repo
REDIS=~/mct/redis-env/bin/redis-server
REDIS_CLI=~/mct/redis-env/bin/redis-cli
PY=~/mct/venv-test/bin/python
PORT_REDIS=7391
PORT_WEB=8391
FIXTURE=~/mct/data/fixtures/wildtrack_7cam.jsonl

export PYTHONPATH=src PYTHONIOENCODING=utf-8
export REDIS_URL=redis://localhost:$PORT_REDIS/0
export MCT_DB_PATH=$HOME/mct/demo/mct.db  # ngoài repo: SQLite không commit
export MCT_HOMOGRAPHY_DIR=configs/cameras/homography/wildtrack
export MCT_TOPOLOGY=configs/demo/wildtrack.topology.yaml

feed() {
  cd "$REPO"
  $PY -m mct --config configs/demo/wildtrack.mct.yaml --db "$MCT_DB_PATH" \
    --source "$FIXTURE" --topology "$MCT_TOPOLOGY" \
    --homography-dir "$MCT_HOMOGRAPHY_DIR" \
    --publish --redis-url "$REDIS_URL" 2>&1 | tail -2
}

case "${1:-start}" in
  start)
    cd "$REPO"
    pgrep -u "$USER" -f "redis-server .*:$PORT_REDIS" >/dev/null 2>&1 || \
      $REDIS --port $PORT_REDIS --save "" --appendonly no --daemonize yes
    sleep 1
    $REDIS_CLI -p $PORT_REDIS ping
    pgrep -u "$USER" -f "uvicorn dashboard.app" >/dev/null 2>&1 || {
      nohup $PY -m uvicorn dashboard.app:app --port $PORT_WEB --log-level warning \
        > /tmp/uvicorn.log 2>&1 &
      sleep 3
    }
    echo "dashboard: http://localhost:$PORT_WEB (trên ut-hpc)"
    curl -s localhost:$PORT_WEB/health | $PY -c \
      "import json,sys; d=json.load(sys.stdin); print('health:', {k: d[k] for k in ('status','redis','calibrated_cameras')})"
    echo "--- phát một lượt dữ liệu (~80s) ---"
    feed
    ;;
  feed) feed ;;
  loop) while true; do feed; sleep 2; done ;;
  status)
    echo "redis:     $(pgrep -u "$USER" -f "redis-server .*:$PORT_REDIS" | wc -l) tiến trình"
    echo "dashboard: $(pgrep -u "$USER" -f 'uvicorn dashboard.app' | wc -l) tiến trình"
    curl -s localhost:$PORT_WEB/health || echo "(chưa bật)"
    ;;
  stop)
    pkill -u "$USER" -f "uvicorn dashboard.app" 2>/dev/null
    $REDIS_CLI -p $PORT_REDIS shutdown nosave 2>/dev/null
    echo "đã tắt dashboard + redis"
    ;;
  *) echo "dùng: start | feed | loop | status | stop"; exit 1 ;;
esac
