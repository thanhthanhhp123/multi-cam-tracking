# 2026-09-06 (phiên 19) — Engine nhanh 4.1 lần (36.8 → 151.9 msg/s), và probe thôi chặn trên Redis

- **Mốc:** M4/M5 (hiệu năng) | **Máy:** máy dev (soạn) + `ut-hpc` (profile + đo) | **Thời lượng:** ~2h, **không tốn GPU**

## Mục tiêu phiên

- Phiên trước phát hiện chỗ nghẽn KHÔNG nằm ở GPU mà ở engine liên kết: 36.8 message/giây,
  trong khi 4 camera ở 25 fps cần 100. Sửa cho đủ nhanh.
- Bỏ vòng gọi Redis đồng bộ nằm trong pad probe của DeepStream.

## Đã làm

**Tối ưu engine — bốn thay đổi, không cái nào đổi kết quả:**

1. **Chiếu cả quỹ đạo bằng một phép nhân ma trận.** `HomographyMapper.project_many()` (mới)
   nhận mảng (n, 2). `affinity._project_path` dò bằng `getattr` và tự lùi về `project()`
   từng điểm khi mapper không có — giao thức `GroundMapper` không đổi nên mapper giả trong
   test vẫn chạy. Trước đó là 7,8 triệu lời gọi `apply_homography`, chiếm 42% thời gian.
2. **Cache quỹ đạo đã chiếu sống qua NHIỀU cửa sổ** thay vì chết theo mỗi `build_cost_matrix`.
   `Associator` giữ cache; mỗi ô lưu `(chính list quỹ đạo, số điểm, kết quả)` và chỉ nhận
   khi `cached_path is path` và số điểm không đổi — giữ tham chiếu mạnh nên `id()` không thể
   bị cấp lại cho list khác, còn số điểm bắt được quỹ đạo đang dài ra.
3. **Bỏ `np.clip` trong `_synchronized_distance`.** `clip` dựng một `finfo` mỗi lời gọi;
   riêng nó tốn 5.4 s trong 32 s. `np.maximum`/`np.minimum` cho cùng kết quả.
   Cộng thêm một phép kiểm giao nhau về thời gian (4 phép so sánh) trước khi đụng
   `searchsorted` trên cả mảng.
4. **`Gallery.find_by_tracklet` tra bảng chỉ mục** thay vì quét toàn bộ gallery —
   2,9 triệu lời gọi `owns_tracklet` cho một lượt chạy. Chỉ mục chỉ giữ tracklet ĐANG được
   sở hữu (bỏ khi bị thay, bỏ khi `prune` đóng track) nên nó lớn theo số track đang mở, không
   theo tổng số tracklet từng thấy.

**`common.streams.QueuedFramePublisher` (mới)** — hàng đợi có trần + luồng nền đẩy Redis theo
lô, thay cho `FramePublisher.publish()` gọi thẳng trong probe. `FramePublisher.publish_many()`
(mới) gửi cả lô bằng một pipeline. `src/ds_pipeline/__main__.py` dùng nó và in thống kê lúc đóng.

**475 passed, 8 skipped**, ruff sạch. 7 test mới cho hàng đợi (chạy bằng publisher giả, không
cần Redis), 1 test cho trần của bảng chỉ mục.

## Quyết định kỹ thuật

**1. Tối ưu hoá phải chứng minh được là KHÔNG đổi kết quả, và đây là cách chứng minh.**
Chạy lại trọn đường online → export → TrackEval sau khi tối ưu: HOTA **16.210**, AssA
**11.131**, IDF1 **20.918**, 433 Global ID — trùng khít từng chữ số với bản trước khi tối ưu.
Một tối ưu hoá làm đổi điểm số dù chỉ ở chữ số thứ ba là một tối ưu hoá đã làm sai gì đó, và
không có phép so này thì không phân biệt được với việc "vô tình chỉnh tham số".

**2. Khoá cache theo `id()` chỉ an toàn khi giữ tham chiếu mạnh.** Cache cũ sống trong đúng
một lần dựng ma trận nên `id(path)` không thể bị cấp lại. Kéo dài tuổi thọ cache thì điều đó
không còn đúng: một list bị thu hồi, list mới trùng địa chỉ, và cache trả về quỹ đạo của
người khác — sai hình học **im lặng**, đúng loại lỗi mà đồ án này đã dính nhiều lần. Nên mỗi ô
giữ luôn tham chiếu tới list và so bằng `is`. Kèm trần 8192 ô, đầy thì xoá sạch: cache chỉ để
tăng tốc, mất nó không đổi kết quả.

**3. Hàng đợi phải CÓ TRẦN, và đầy thì bỏ khung CŨ nhất.** Hàng đợi không trần chỉ đổi một chỗ
hỏng ồn ào (khung hình đứng vì probe chặn) lấy một chỗ hỏng im lặng (RAM phình tới khi bị
kill). Có trần thì biết mình mất gì và mất bao nhiêu. Bỏ cái cũ nhất vì hệ thống trả lời
"người đó đang ở đâu": đã tụt lại thì dữ liệu tươi có giá hơn, và mất một khung chỉ tạo một lỗ
trong quỹ đạo — thứ `TrackletBuilder` vốn đã phải chịu được với detector recall 44.9%.
Số khung rơi được đếm và log định kỳ chứ không log từng cái, vì log mỗi lần rơi sẽ thành chỗ
nghẽn mới.

**4. Luồng nền bắt `Exception` rộng, có chủ ý.** Một lô hỏng mà giết luồng nền thì mất im lặng
toàn bộ luồng dữ liệu từ pipeline sang engine — tệ hơn nhiều so với bỏ một lô rồi chạy tiếp.
Đếm vào `n_failed` và log kèm traceback.

## Số liệu đo được

**Cấu hình:** fixture `ds_wildtrack_7cam.jsonl` (2800 message, 7 camera, ~15 detection/message,
313 GlobalTrack mở lúc cuối), config `wildtrack_ds.mct.yaml` + homography 7 camera, chạy trên
head node `ut-hpc` (Python 3.10.12, `venv-test`). Profile bằng `cProfile` trên đúng vòng
`Engine.feed()`.

### Thông lượng engine

| bước | msg/s | so với mốc cần (100 msg/s = 4 camera × 25 fps) |
|---|---|---|
| trước phiên 19 | **36.8** | 0.37× |
| + chiếu theo lô + cache qua cửa sổ | 87.1 | 0.87× |
| + bỏ `np.clip` + kiểm giao thời gian + chỉ mục | **151.9** | **1.52×** |

**Nhanh 4.1 lần.** Thời gian một lượt 2800 message: 76.2 s → 18.4 s.

### Thời gian đi đâu (cumulative, giây trên tổng)

| | trước | sau bước 1–2 |
|---|---|---|
| tổng vòng gán | 75.9 | 31.9 |
| `_ground_term` | 62.5 (82%) | 22.9 (72%) |
| `_world_path` | 46.8 (62%) | — (cache trúng) |
| `apply_homography` (7,8 triệu lời gọi) | 31.7 (42%) | — (chiếu theo lô) |
| `_synchronized_distance` | 18.2 | 17.6 → chỗ nóng tiếp theo |

Trong `_synchronized_distance`, `np.clip` một mình chiếm 5.4 s vì mỗi lời gọi dựng một
`finfo` (1,1 triệu lần gọi `getlimits.__init__`).

### Chấm lại để chứng minh không đổi hành vi

| | trước tối ưu | sau tối ưu |
|---|---|---|
| HOTA | 16.210 | **16.210** |
| DetA / AssA | 24.123 / 11.131 | **24.123 / 11.131** |
| IDF1 | 20.918 | **20.918** |
| Global ID sinh ra | 433 | **433** |

## Vướng mắc / chưa xong

- **151.9 msg/s là đo trên head node dùng chung của `ut-hpc`**, một tiến trình, không có
  Redis thật trong vòng lặp. Máy demo sẽ khác; con số cần đo lại tại chỗ.
- Margin so với 100 msg/s chỉ còn **1.5 lần**, trong khi phía GPU dư 7 lần. Chi phí engine
  tỉ lệ với số tracklet trong cửa sổ nhân số GlobalTrack đang mở, nên đông người hoặc chạy
  lâu (gallery phình) sẽ ăn margin đó. `_synchronized_distance` là chỗ nóng tiếp theo nếu
  cần thêm.
- **Hàng đợi mới chưa chạy trên phần cứng thật** — mọi test dùng publisher giả. Cần một lượt
  `--publish` trên máy GPU để biết `maxsize=2000` và `batch=32` có hợp lý không, và đọc
  `n_dropped` sau khi chạy.
- Trần hàng đợi 2000 khung ≈ 20 giây dữ liệu ở 4 camera × 25 fps. Chưa có cơ sở đo đạc cho
  con số đó, chỉ là ước lượng.

## Bước tiếp theo

1. Đếm phần dư của 433 Global ID theo danh tính (nợ từ phiên 18).
2. Chạy `--publish` trên máy GPU, đọc `n_dropped` và `max_depth` để chốt trần hàng đợi.
3. Chấm `onnx_gtbox` bằng HOTA (nợ từ phiên 15).
