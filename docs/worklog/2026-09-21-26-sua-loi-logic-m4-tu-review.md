# 2026-09-21 — Sửa lỗi logic M4 từ một lượt review độc lập

- **Mốc:** M4 (module liên kết đa camera) | **Máy:** máy dev (macOS, `.venv` Python 3.11) | **Thời lượng:** ~1h

## Mục tiêu phiên

- Một lượt review độc lập của GPT trên M4 nêu 5 lỗi logic. Kiểm từng cái bằng repro chứ không tin mô tả.
- Sửa những cái đúng, mỗi cái kèm test bắt được lỗi.

## Đã làm

Repro trước (script ở scratchpad, không commit) — cả 5 điều GPT nói đều đúng về mặt code, nhưng
mức độ nghiêm trọng khác nhau (xem bảng).

| # | Lỗi | Mức thật | Trạng thái |
|---|---|---|---|
| 1 | Khởi động lại → `tracklet_id`/`global_id` cấp lại từ 1, upsert đè lịch sử SQLite | **P0** | Đã sửa |
| 2 | Hai mảnh nối tiếp cùng camera lọt vào cùng một `assign()` → hai Global ID | P1, hẹp | Đã sửa |
| 3a | `start_ms`/`first_bbox`/`first_ground_point` lấy từ message TỚI đầu tiên, không phải SỚM nhất | P1 | Đã sửa |
| 3b | `close_expired()` dùng watermark chung mọi camera | P2, chủ ý | KHÔNG sửa |
| 4 | `ts_ms` lấy từ probe chứ không phải `ntp_timestamp` | P2, đã đo | Chỉ sửa docstring |
| 5 | Guard kích thước khung hình của homography không ai gọi | P2, tiềm ẩn | Đã sửa |

Repro P0: chạy hai lần cùng DB → DB chỉ còn 1 dòng `(gid 1, cam01, start 1000, end 9330)`; tracklet
cam02 của lần 2 biến mất và `end_ms` của nó chui vào dòng cam01. Repro #2: hai mảnh không chồng thời
gian trong một vòng cho GID 1 và 2; tách thành hai vòng thì mảnh 2 ghép đúng vào GID 1.

File đã sửa:
- `src/mct/store.py` — `Store.next_ids()`: `(tracklet_id, global_id)` nhỏ nhất chưa dùng.
- `src/mct/tracklet.py`, `src/mct/gallery.py` — `reserve_ids_from()` (chỉ tăng, không kéo lùi);
  `Tracklet.add` cập nhật mốc đầu khi gặp `ts_ms` sớm hơn.
- `src/mct/__main__.py` — `Engine.__init__` nhảy qua ID đã dùng trong DB; `Engine._check_frame_size`
  gọi `warn_frame_sizes` một lần cho mỗi camera, kết quả nằm ở `engine.size_warnings`.
- `src/mct/associator.py` — `_match_one_camera` xử lý hàng chưa ghép theo `start_ms`, mỗi hàng thử
  khớp với track vừa tạo trong CÙNG vòng trước khi tự tạo track (`_match_created`).
- `src/ds_pipeline/probes.py` — chỉ docstring/comment.
- Test mới: `test_tracklet.py` (2), `test_associator.py` (2), `test_store.py` (1),
  `test_engine_online.py` (2). Gỡ phần sửa trong `src/` thì 7 test đỏ; test đối chứng
  "mảnh nối tiếp nhưng khác người vẫn tách" xanh cả hai chiều.
- **Đổi một assertion cũ:** `test_message_lech_thu_tu_khong_lam_lui_moc_ket_thuc` từng ghim
  `start_ms == 1_000` khi có khung ở t=900 tới muộn (span `1000..1100` không chứa khung đó). Đổi
  thành 900. Không có quyết định nào trong worklog bảo vệ hành vi cũ.

## Quyết định kỹ thuật

- **#1: chặn ghi đè bằng cách nhảy ID, không hồi phục gallery.** SQLite không lưu embedding nên
  không dựng lại gallery được — sau restart, người của lần chạy trước chắc chắn nhận Global ID mới.
  Phương án loại: đổi khoá `appearances` thành `(run_id, tracklet_id)` (đổi schema, đụng dashboard);
  sửa `ON CONFLICT` để cập nhật đủ cột (che triệu chứng, dòng vẫn bị đè). Nhảy ID là 6 dòng và giữ
  nguyên schema.
- **#2: tham lam theo `start_ms`, chỉ so với track MỚI tạo trong vòng.** Không chạy lại Hungarian
  trên toàn bộ track mở: làm vậy đổi luôn kết quả những hàng "bị lấy mất" (`taken`) vốn đúng, mà
  không có fixture DeepStream thật ở máy này để đo HOTA trước/sau. Sửa hẹp thì chỉ đổi kết quả đúng
  ở chỗ lỗi. Thống kê `no_candidate`/`rejected_by_threshold` giữ nguyên nghĩa (lý do theo ảnh chụp
  đầu vòng).
- **#3b giữ nguyên:** watermark chung là chủ ý — camera chết hẳn thì watermark theo từng camera sẽ
  không bao giờ đóng tracklet của nó. Chỉ dính khi một camera trễ hơn `idle_timeout_ms` so với
  camera khác. Đánh đổi phải ghi vào chương 6 nếu chạy RTSP thật.
- **#4 chỉ sửa docstring:** docstring cũ nói "lấy từ `ntp_timestamp`" nhưng code dùng giờ probe.
  Worklog 2026-09-18-23 đã đo `t1−t0` = 67.4 ms (p99 69.4), đồng đều giữa camera, nên so với
  `min_ms` 3000 và `ground_time_tol_ms` 400 là nhỏ. Đổi sang `t0` cần kiểm trên `vast-gpu` và làm
  fixture cũ lệch mốc — để dành cho lần chạy RTSP thật.
- **#5: gọi guard với `msg.frame_width/height` là ĐÚNG không gian toạ độ**, vì probe đã scale bbox
  về `frame_width × frame_height` (= kích thước mux). Hiện chưa gây lỗi vì nguồn 1080p, mux 1920×1080.
  `GroundMapper` là Protocol không khai `warn_frame_sizes` nên engine dò bằng `getattr`.

## Số liệu đo được

— (không đo hiệu năng/độ chính xác). Kết quả test: 571 passed, 8 skipped, `ruff check` và
`ruff format --check` sạch.

## Vướng mắc / chưa xong

- **Chưa đo ảnh hưởng của #2 lên HOTA/IDF1** — fixture DeepStream thật (`ds_wildtrack_7cam*`) không
  có ở máy này; `two_cam_walk` (tổng hợp) có 0 cặp mảnh nối tiếp cùng batch. Số F1 0.929/0.765 trong
  CLAUDE.md §6 là trước bản sửa này; cần chạy lại khi có fixture.
- `tests/test_dashboard.py` không chạy được ở `.venv` này (starlette đòi `httpx2`) — lỗi môi trường,
  không liên quan bản sửa (chỉ thêm `Store.next_ids`). Chưa chạy lại trên venv 3.10 đúng chuẩn;
  `uv` chưa cài trên máy macOS này nên toàn bộ phiên chạy bằng Python 3.11 — quy tắc bất biến số 4
  chưa được kiểm bằng 3.10.
- `Tracklet.ground_path` vẫn append theo thứ tự tới. `affinity` sắp lại khi chiếu/lọc, nhưng
  `_edge_point` với `ground_smooth=none` lấy `path[0]`/`path[-1]` theo thứ tự tới — chỉ sai khi
  message lệch thứ tự VÀ tắt lọc. Chưa sửa, chưa đo.
- Restart vẫn mất danh tính người đang ở trong gallery (giới hạn của thiết kế, xem #1).

## Bước tiếp theo

1. Khi có fixture DeepStream thật: chạy `eval/compare_online_offline.py` trước/sau (`git stash` phần
   `src/mct/associator.py`) để xem #2 có đổi HOTA/AssA không.
2. Dựng venv 3.10 (`uv`) trên máy macOS và chạy lại suite, kể cả `test_dashboard`.
3. Nếu chạy RTSP thật: đo `t1−t0` theo từng camera; nếu hết đồng đều thì đổi `ts_ms` sang `t0`.
