# 2026-08-27 — Khởi tạo dự án, chốt kiến trúc tổng thể

- **Mốc:** trước M0 | **Máy:** Mac M1 | **Thời lượng:** ~1h

## Mục tiêu phiên

- Đọc và hiểu đề cương đồ án.
- Chốt kiến trúc, cấu trúc source code, hướng xử lý.
- Tạo `CLAUDE.md` làm tài liệu nền cho các phiên sau.

## Đã làm

- Đọc `docs/DoAn_MultiCameraTracking_DeepStream.docx` (7 chương) — bài toán MTMCT 3–4 camera,
  DeepStream + YOLO + Re-ID, đóng góp chính là module liên kết đa camera.
- Khảo sát môi trường máy dev: **Mac M1 (arm64), không có GPU NVIDIA**, Docker 28.1.1,
  Python 3.11.2, có conda. → phát hiện ràng buộc chi phối toàn bộ thiết kế (xem mục dưới).
- Tạo `CLAUDE.md` — kiến trúc, contract dữ liệu, thuật toán association, quy ước code,
  lộ trình M0–M7, cạm bẫy đã biết.
- Tạo `.gitignore` (loại trừ `data/`, `models/`, `*.engine`, `*.onnx`, video — dung lượng lớn
  và có ràng buộc quyền riêng tư theo đề cương mục 4.3.3), `git init`.
- Tạo `docs/worklog/` + quy ước ghi nhật ký.

## Quyết định kỹ thuật

**1. Tách codebase làm hai nửa, ranh giới là Redis Streams.** *(quyết định lớn nhất phiên này)*

DeepStream không chạy được trên macOS/M1, nhưng máy dev chính là Mac. Nếu để pipeline và
engine liên kết trong cùng một process thì mọi việc phát triển đều phải ngồi cạnh máy GPU.

Chọn: pipeline DeepStream publish metadata lên Redis Stream `mct:frames`; engine liên kết là
process Python riêng, consume từ đó. Nhờ vậy **ghi lại luồng metadata thật một lần trên máy GPU,
rồi phát lại trên Mac** để phát triển module liên kết — vốn là đóng góp chính của đồ án
(tuần 11–13) và là phần cần sweep tham số nhiều nhất.

Phương án loại:
- *Kafka qua `nvmsgconv`/`nvmsgbroker`* — đúng "chuẩn DeepStream", viết vào báo cáo đẹp hơn,
  nhưng nặng (Zookeeper + Kafka), tốn RAM, khó debug. Không tương xứng với quy mô 3–4 camera.
- *ZeroMQ PUB/SUB* — nhẹ nhất nhưng không có persistence → không replay được, mất đúng
  lợi ích chính đang cần.
- *Cùng một process* — đơn giản nhất nhưng buộc mọi thứ chạy trên máy GPU. Loại vì lý do trên.

Hệ quả: đặt ra **quy tắc bất biến** — `src/common`, `src/mct`, `src/dashboard`, `src/tools`
tuyệt đối không `import pyds`/`gi`/`tensorrt`/`cuda`. Sẽ có test tự động canh
(`tests/test_no_gpu_imports.py`). Vi phạm quy tắc này là hỏng cả quy trình dev.

**2. Đảo thứ tự M3 và M4 so với trình tự tự nhiên.** Ghi fixture Re-ID thật (M3, máy GPU)
đặt trước khi phát triển engine liên kết (M4, Mac). Có fixture rồi thì phần khó nhất làm offline được.
Nếu máy GPU chưa sẵn sàng lúc bắt đầu, M0 vẫn chạy được bằng fixture tổng hợp sinh tay.

**3. Chốt contract dữ liệu sớm**, vì đây là chỗ hay vỡ nhất khi hai nửa phát triển độc lập:
- bbox theo **độ phân giải camera gốc**, không phải toạ độ `nvstreammux` — DeepStream trả
  `rect_params` theo toạ độ muxer, probe phải scale ngược. Bug kinh điển, ghi rõ trong CLAUDE.md.
- giữ **cả** `frame_pts_ns` (PTS GStreamer) và `ts_ms` (wall clock NTP); ràng buộc thời gian
  di chuyển xuyên camera dùng `ts_ms`.
- embedding L2-normalize **tại producer**, msgpack raw float32 trên wire (JSON phình ~3x và
  mất độ chính xác), base64/JSONL cho fixture để soạn tay được khi viết test.
- `cam_id` là string ổn định, **không dùng index streammux** — index đổi khi bật/tắt camera.

**4. Nhắm Python 3.10**, không phải 3.11 của Mac — để khớp phiên bản trong container
DeepStream 7.x / Ubuntu 22.04, tránh code dùng chung chạy được trên Mac mà vỡ trên máy GPU.

**5. Thuật toán liên kết**: query embedding = trung bình top-k theo confidence (không dùng
embedding frame cuối, hay dính crop mờ/bị che) → lọc ứng viên bằng topology + transit time →
cost `1 − cosine` (+ `λ·d_ground` qua homography cho cặp overlap) → Hungarian → ngưỡng τ
hoặc tạo Global ID mới. Giữ **cả hai chế độ** `online` (sản phẩm bàn giao) và `offline`
(Hungarian theo lô, cho cận trên độ chính xác) để báo cáo định lượng được cái giá của real-time.

## Số liệu đo được

— (chưa có gì để đo)

## Vướng mắc / chưa xong

- **Chưa pin phiên bản DeepStream.** Định hướng 7.x / Ubuntu 22.04 / CUDA 12.x / TensorRT 10.x,
  nhưng DeepStream rất kén cặp driver–CUDA–TensorRT. Phải xem driver trên máy GPU thật rồi
  mới chốt vào `.env` + Dockerfile. Đề cương mục 5.2 cũng xếp đây là rủi ro số 1.
- **Chưa chốt đường lấy Re-ID embedding.** Hai khả năng:
  (A) ReID extractor tích hợp sẵn trong `nvtracker` (NvDCF/NvDeepSORT) — hiệu quả hơn vì tracker
  đã crop sẵn; (B) SGIE `nvinfer` thứ hai chạy OSNet trên crop — đúng như đề cương mục 3.3 mô tả.
  Nghiêng về (A), nhưng **tên meta type khác nhau giữa các phiên bản DeepStream** →
  phải tra tài liệu bản đã cài, không đoán.
- Chưa có máy GPU trong tay tại phiên này; toàn bộ M1–M3 đang chờ.
- Chưa khảo sát camera IP thực tế (model nào, RTSP URL ra sao) và chưa có sơ đồ bố trí.

## Bước tiếp theo

Bắt đầu **M0** — làm được 100% trên Mac, không cần chờ máy GPU:

1. `pyproject.toml` + `Makefile` + khung thư mục `src/`.
2. `src/common/schema.py` — dataclass `Detection`/`FrameMessage` + encode/decode msgpack & JSONL.
3. `src/common/streams.py` — wrapper Redis Streams (producer + consumer group).
4. `src/tools/replay_metadata.py` và sinh fixture tổng hợp 2 camera cho `tests/fixtures/`.
5. `tests/test_no_gpu_imports.py` — canh quy tắc bất biến số 1.
