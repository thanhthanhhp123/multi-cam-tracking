# Theo dõi đối tượng đa camera trên nền NVIDIA DeepStream

Đồ án tốt nghiệp — hệ thống MTMCT (Multi-Target Multi-Camera Tracking): theo dõi người
qua 3–4 camera IP và gán **Global ID** nhất quán khi họ di chuyển xuyên camera, kể cả
giữa các camera không có vùng nhìn chồng lấn.

- Đề cương đầy đủ: [`docs/DoAn_MultiCameraTracking_DeepStream.docx`](docs/DoAn_MultiCameraTracking_DeepStream.docx)
- Hướng dẫn kỹ thuật cho người (và Claude) làm việc trong repo: [`CLAUDE.md`](CLAUDE.md)
- Nhật ký từng phiên làm việc: [`docs/worklog/`](docs/worklog/)

## Kiến trúc

Ba tầng, tách rời qua Redis Streams:

```
camera IP --RTSP--> [ pipeline DeepStream ]  máy Ubuntu + GPU NVIDIA
                              |  bbox + local track ID + Re-ID embedding
                              v
                     [ Redis Streams ]       <-- ranh giới duy nhất
                              |
                              v
                     [ engine liên kết ]     chạy được trên máy không GPU
                              |  Global ID + trajectory
                              v
                     [ SQLite + dashboard ]
```

Ranh giới Redis Streams là quyết định thiết kế trung tâm: nó cho phép ghi lại luồng
metadata thật từ máy GPU một lần, rồi phát lại trên máy dev để phát triển engine liên
kết — phần chiếm nhiều thời gian nhất và là đóng góp kỹ thuật chính của đồ án.

## Yêu cầu môi trường

| | Máy dev | Máy chạy pipeline |
|---|---|---|
| Hệ điều hành | bất kỳ (đang dùng macOS) | Ubuntu 22.04 |
| Phần cứng | không cần GPU | GPU NVIDIA rời, ≥8GB VRAM |
| Chạy được | mọi thứ trừ `src/ds_pipeline` | tất cả |

DeepStream không chạy trên macOS. Mọi package ngoài `src/ds_pipeline` được thiết kế để
chạy không cần GPU, và có test tự động canh điều đó.

## Bắt đầu

```bash
make dev        # tạo .venv + cài dependencies
make test       # test không cần GPU (test cần Redis tự bỏ qua nếu chưa `make up`)
make lint
```

Chạy thử luồng dữ liệu đầy đủ mà không cần camera hay GPU:

```bash
make up                    # bật Redis
make fixture               # sinh metadata giả lập 2 camera + ground-truth
make replay                # phát vào Redis theo đúng nhịp thời gian gốc
```

Kiểm tra bằng dashboard:

```bash
.venv/bin/uvicorn dashboard.app:app --port 8000
open http://127.0.0.1:8000/health
```

`make help` liệt kê toàn bộ target.

## Trạng thái

Xem bảng lộ trình M0–M7 ở [`CLAUDE.md` §9](CLAUDE.md). Hiện tại: **M0 xong** — khung
repo, contract dữ liệu, wrapper Redis, bộ sinh fixture, dashboard rỗng. Đang kéo M4
(engine liên kết) lên sớm bằng fixture từ WildTrack — dữ liệu multi-camera thật, có
ground-truth Global ID (xem `tests/fixtures/README.md`). M1–M3 (pipeline DeepStream)
chạy song song khi thuê được máy GPU.
