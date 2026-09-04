.PHONY: help dev test lint fmt up down replay record fixture wildtrack-annotations \
        wildtrack-fixture wildtrack-homography engine engine-fixture dashboard compare eval \
        ds-build ds-run clean
.DEFAULT_GOAL := help

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
COMPOSE := docker compose -f docker/compose.yml

FIXTURE ?= tests/fixtures/two_cam_walk.jsonl
OUT     ?= tests/fixtures/recorded.jsonl

help:  ## Liệt kê các target
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(VENV):
	python3 -m venv $(VENV)

# ---------- Chạy được trên Mac, không cần GPU ----------

dev: $(VENV)  ## Tạo venv + cài dependencies phía CPU (editable)
	$(PIP) install -q -U pip
	$(PIP) install -q -e ".[dev]"
	@echo "Xong. Kích hoạt bằng: source $(VENV)/bin/activate"

test:  ## Chạy pytest (tự bỏ qua test có mark gpu)
	$(VENV)/bin/pytest

lint:  ## ruff check + kiểm tra format  (LƯU Ý: chỗ chạy chuẩn là ut-hpc, xem CLAUDE.md §2)
	$(VENV)/bin/ruff check src tests eval
	$(VENV)/bin/ruff format --check src tests eval

fmt:  ## Tự động format và sửa lỗi lint sửa được
	$(VENV)/bin/ruff format src tests eval
	$(VENV)/bin/ruff check --fix src tests eval

up:  ## Bật Redis (+ engine, dashboard khi đã có)
	$(COMPOSE) up -d

down:  ## Tắt các service
	$(COMPOSE) down

fixture:  ## Sinh lại fixture tổng hợp 2 camera
	$(PY) -m tools.make_synthetic_fixture --out $(FIXTURE)

WILDTRACK_DIR ?= data/wildtrack
REID_ONNX     ?= models/reid/osnet_x1_0_market1501.onnx

wildtrack-annotations:  ## Tải annotation + calibration WildTrack (nhỏ; ảnh gốc tải riêng)
	$(PY) -m tools.fetch_wildtrack_annotations --dest $(WILDTRACK_DIR)

wildtrack-fixture:  ## WildTrack -> fixture thật (cần $(WILDTRACK_DIR)/Image_subsets + [reid])
	$(PY) -m tools.wildtrack_to_fixture --wildtrack-dir $(WILDTRACK_DIR) \
		--views 1,4,7 --reid-onnx $(REID_ONNX) --out tests/fixtures/wildtrack_3cam.jsonl

wildtrack-homography:  ## Hiệu chỉnh homography 7 camera WildTrack từ chú thích
	$(PY) -m tools.calibrate_homography --wildtrack-dir $(WILDTRACK_DIR) \
		--out configs/cameras/homography/wildtrack

DB ?= data/mct.db

engine:  ## Chạy engine liên kết (Redis -> Global ID -> mct:global + SQLite)
	$(PY) -m mct --config configs/mct.yaml --db $(DB) --publish

engine-fixture:  ## Chạy engine trên fixture, không cần Redis — make engine-fixture FIXTURE=...
	$(PY) -m mct --config configs/mct.yaml --source $(FIXTURE) --db $(DB)

dashboard:  ## Mở dashboard ở http://localhost:8000 (đọc Redis + $(DB))
	MCT_DB_PATH=$(DB) $(VENV)/bin/uvicorn dashboard.app:app --port 8000 --reload

compare:  ## Đo chênh lệch online vs offline trên fixture — make compare FIXTURE=...
	$(PY) -m eval.compare_online_offline --fixture $(FIXTURE) \
		--homography-dir configs/cameras/homography/wildtrack

replay:  ## Phát lại fixture vào Redis — make replay FIXTURE=...
	$(PY) -m tools.replay_metadata --fixture $(FIXTURE)

record:  ## Ghi Redis stream ra fixture — make record OUT=...
	$(PY) -m tools.record_metadata --out $(OUT)

eval:  ## Chạy TrackEval trên kết quả trong eval/   (M6)
	$(PY) eval/run_trackeval.py

clean:  ## Xoá venv và cache
	rm -rf $(VENV) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# ---------- Chỉ trên máy Ubuntu + GPU NVIDIA ----------

ds-build:  ## Build image DeepStream
	docker build -f docker/deepstream.Dockerfile -t mct-deepstream:dev .

ds-run:  ## Chạy pipeline theo configs/pipeline/streams.yaml
	docker compose -f docker/compose.yml -f docker/compose.gpu.yml up ds-pipeline
