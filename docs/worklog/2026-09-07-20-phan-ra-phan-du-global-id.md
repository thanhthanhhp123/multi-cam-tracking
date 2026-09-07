# 2026-09-07 (phiên 20) — Phân rã 433 Global ID: một nửa là rác, và "nghiêng về tách" chỉ đúng một nửa

- **Mốc:** M4 (đóng góp chính) | **Máy:** máy dev (soạn) + `ut-hpc` (chạy + đo) | **Thời lượng:** ~1.5h, **không tốn GPU**

## Mục tiêu phiên

- Việc số 1 của phiên 19 (nợ từ phiên 18): **đếm phần dư của 433 Global ID theo danh tính**.
  Phiên 18 chốt `max_cost: 0.90` dựa trên lập luận "433 > 313 danh tính nghĩa là nghiêng về
  TÁCH thay vì GỘP, mà tách thì còn sửa được". Lập luận đó dựa trên đúng một con số và chưa
  ai kiểm.

## Đã làm

- **`eval/diagnose_global_ids.py`** (mới) — phân rã tổng số Global ID thành một đẳng thức
  kiểm tra được, đọc từ bảng `appearances` của SQLite store (chính nguồn mà
  `tools/export_trackeval.py` dùng để xuất điểm) ghép với bảng `.gt.json`. Năm mục: phân rã,
  vỡ theo danh tính, khối lượng của mảnh dư, gộp nhầm, và lý do mỗi Global ID ra đời.
- **`mct.associator.reason_kind()`** (mới) + `Assignment.reason` giờ mang tiền tố
  `"<loại>: "`. Bảng `appearances` chỉ có một cột `reason` dạng chuỗi tiếng Việt tự do;
  không có tiền tố máy đọc được thì hậu kiểm buộc phải dò chuỗi — đúng thứ mà docstring của
  `_why_new` đã cấm từ đầu.
- Sửa `src/common/streams.py` cho khớp `ruff format` (một dòng lệch từ phiên 19 —
  `make lint` trên `ut-hpc` bắt được).
- **496 passed, 8 skipped**, ruff sạch. 21 test mới, dựng kịch bản có đáp án đếm được bằng
  tay: một người liền mạch, một người xé ba mảnh, một Global ID ôm hai người, một Global ID
  toàn tracklet không nhãn, và cả trường hợp trộn cả ba.

## Quyết định kỹ thuật

**1. Phân rã phải là một ĐẲNG THỨC, không phải một bộ số rời rạc.** Công cụ chốt lại:

```
n_gid = n_rác + n_danh_tính_phủ + n_mảnh_dư − n_chồng_do_gộp
```

Bốn đại lượng vế phải đếm độc lập nhau, nên đẳng thức vừa là kết quả vừa là phép tự kiểm —
lệch một đơn vị là một trong bốn phép đếm sai, và `decompose()` ném lỗi ngay thay vì in ra
một bảng đẹp đẽ mà sai. Phương án bị loại: báo cáo ba con số "số ID vỡ / số ID gộp / số ID
rác" theo kiểu liệt kê. Chúng chồng lấn nhau (một Global ID vừa là mảnh dư của người này vừa
lẫn người kia) nên cộng lại không ra tổng, và không có cách nào phát hiện khi một phép đếm
sai.

**2. "Mảnh chính" chọn theo SỐ KHUNG, không theo số tracklet.** Với mỗi danh tính, mảnh
chính là Global ID giữ nhiều khung nhất của người đó; phần còn lại là mảnh dư. Chọn theo số
tracklet thì 5 mảnh vụn 3 khung sẽ thắng 1 tracklet 100 khung, mà chỗ hệ thống thật sự nhận
ra một người là chỗ giữ phần lớn thời lượng của người đó.

**3. Đo phần dư bằng KHỐI LƯỢNG, không chỉ bằng số đếm.** Số mảnh dư một mình không nói gì:
một mảnh 3 khung gần như không ăn vào AssA, còn một người bị chia đôi 50/50 thì mất nửa số
cặp liên kết đúng. Nên mục 3 báo cáo **tỉ lệ khung nằm ngoài mảnh chính** — đại lượng tương
ứng trực tiếp với thứ AssA đo.

**4. Dòng "khai sinh" của một Global ID là dòng CÓ `reason`, không phải dòng sớm nhất.**
Bản đầu lấy theo `start_ms` và cho ra 40 Global ID "không rõ vì sao ra đời". Nguyên nhân:
một tracklet bắt đầu sớm nhưng đóng muộn có thể được GHÉP vào Global ID đã tồn tại, và dòng
rỗng lý do của nó che mất dòng thật nằm phía dưới. Chỉ tracklet nhận ID mới mới có `reason`,
và `ON CONFLICT` không đụng cột đó, nên lọc theo `reason` khác rỗng mới đúng. Sửa xong: 40 →
22 ca không xác định (phần còn lại là ca thật sự mất dấu — người khai sinh đã bị gán sang
track khác, và `global_id` thì `ON CONFLICT` CÓ cập nhật). Ghi lại vì đây là kiểu lỗi hậu
kiểm im lặng giống hệt lỗi `GlobalIdIndex` của phiên 17.

**5. Lập luận "nghiêng về tách" của phiên 18 phải sửa lại, nhưng KHÔNG kéo theo đổi
`max_cost`.** Số đo cho thấy engine gộp nhầm ở mức không hề nhỏ: 24.1% Global ID có nhãn ôm
từ 2 danh tính trở lên, chiếm 13.9% số khung. Tức 433 > 313 không chứng minh được "chỉ tách
chứ không gộp" — nó chỉ là hiệu của hai hiện tượng xảy ra đồng thời. Dù vậy vẫn giữ 0.90:
phần vỡ (43.7% số khung) vẫn lớn gấp ba phần gộp (13.9%), nên hướng đánh đổi mà phiên 18
chọn đúng chiều; chỉ có lý do đưa ra là chưa đủ. Đây là điều chỉnh lập luận, không phải đảo
quyết định.

## Số liệu đo được

**Cấu hình:** fixture `ds_wildtrack_7cam.jsonl` (WildTrack 7 camera, 2 fps, 2800 message —
hộp YOLO11s, `local_track_id` của NvDCF, embedding ReID của nvtracker), config
`configs/demo/wildtrack_ds.mct.yaml` + `wildtrack.topology.yaml` + homography 7 camera,
đường online đầy đủ (`python -m mct --source`). Head node `ut-hpc`, Python 3.10.12,
`venv-test`. Một lượt: **14.8 s** cho 2800 message (phiên 19 đo 18.4 s), **433 Global ID —
trùng khít bản của phiên 18/19**, nên số dưới đây nói về đúng lần chạy đã cho HOTA 16.210.

### 1. Phân rã 433 Global ID

| | số ID | % |
|---|---|---|
| **RÁC** — không một tracklet nào tra được nhãn | **217** | **50.1%** (52.0% số khung) |
| CÓ NHÃN | 216 | 49.9% |
|   ├─ danh tính phủ được | 146 | |
|   ├─ mảnh DƯ do vỡ | +133 | |
|   └─ chồng do GỘP | −63 | |

`433 = 217 + 146 + 133 − 63` ✓

**Một nửa số Global ID không ứng với danh tính nào trong bảng chấm.** Đây là con số quan
trọng nhất của phiên: phần dư 287 ID so với 146 danh tính phủ được thì 217 là rác và chỉ 70
là do vỡ. Mọi nỗ lực chỉnh ngưỡng liên kết chỉ động được vào 70 cái sau.

### 2. Vỡ — một danh tính thành mấy Global ID

| số Global ID | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| số danh tính | **80** | 36 | 14 | 8 | 2 | 3 | 1 | 1 | 1 |

**54.8% danh tính được liên kết ĐÚNG thành một Global ID duy nhất** (80/146), trung bình
1.91 Global ID/danh tính. 77/146 danh tính thấy ở ≥2 camera.

### 3. Mảnh dư to cỡ nào

| | |
|---|---|
| mảnh dư (cặp danh tính × Global ID) | 133, trong đó **91.0% chỉ gồm 1 tracklet** |
| khung nằm ngoài mảnh chính | **7236/16559 = 43.7%** ← phần AssA mất vì vỡ |
| khung/mảnh dư (p25 / p50 / p75 / p95) | 8 / 16 / 48 / 246 |
| % khung ở mảnh chính (p5 / p25 / p50) | 40.6 / 61.5 / **100.0** |

Trung vị 100% nghĩa là quá nửa số danh tính không mất khung nào; phần mất tập trung vào
đuôi dưới. Mảnh dư trung vị 16 khung — nhỏ, nhưng đuôi p95 246 khung thì không.

### 4. Gộp — một Global ID ôm mấy danh tính

| số danh tính | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| số Global ID | 164 | 42 | 9 | 1 |

**52/216 (24.1%) Global ID có nhãn bị lẫn ≥2 người**, chiếm **2300/16559 = 13.9%** số khung
← phần AssA mất vì gộp. Ca nặng nhất: Global ID 45 ôm 3 danh tính trên 717 khung.

### 5. Global ID ra đời vì lý do gì (đếm theo Global ID)

| lý do | mảnh chính | mảnh dư | rác |
|---|---|---|---|
| `threshold` — có ứng viên, ngoại hình không đủ giống | 101 | **61** | 156 |
| `no_candidate` — ràng buộc không–thời gian loại sạch | 22 | 12 | 48 |
| `taken` — ứng viên rẻ nhất bị tracklet khác lấy | 4 | 1 | 3 |
| `empty` — gallery rỗng | 1 | 1 | 1 |
| không xác định (người khai sinh đã bị gán đi nơi khác) | 5 | 8 | 9 |
| **TỔNG** | **133** | **83** | **217** |

**73% mảnh dư (61/83) sinh ra vì `threshold`, không phải vì hình học.** Nút thắt của bước
liên kết vẫn là **ngoại hình**: ràng buộc không–thời gian chỉ chịu trách nhiệm cho 12/83.
Trùng khớp với thống kê của chính engine (336 bị ngưỡng loại so với 85 hết ứng viên) và với
kết luận phiên 12 ("nút thắt chuyển sang ngoại hình").

## Vướng mắc / chưa xong

- **"Rác" không đồng nghĩa với "nhiễu".** Bảng `.gt.json` chỉ gán nhãn được **345/861
  (40.1%)** local track mà pipeline sinh ra: `ds_wildtrack_gt.py` loại thẳng track không đủ
  thuần khiết hoặc khớp quá ít khung thay vì gán bừa. Nên 217 Global ID "rác" trộn hai thứ
  đòi hai cách sửa khác nhau: hộp detector báo nhầm (sửa ở detector) và track có thật nhưng
  chưa chấm được (sửa ở bộ gán nhãn). Chưa tách được hai phần này — cần đối chiếu trực tiếp
  với chú thích WildTrack thay vì chỉ qua bảng `.gt.json`.
- Bảng `.gt.json` phủ **146/313 danh tính (46.6%)**, trong khi TrackEval chấm trên toàn bộ
  313. Vì vậy tỉ lệ ở mục 2–4 là tỉ lệ **trên phần chấm được**, không phải trên toàn hệ
  thống; đừng ghép thẳng chúng cạnh HOTA trong chương 6 mà không nói rõ mẫu số.
- 22 Global ID không truy được lý do khai sinh (5.1%) — giới hạn của việc bảng `appearances`
  cập nhật `global_id` nhưng không cập nhật `reason`. Không đáng đổi schema để sửa.

## Bước tiếp theo

1. **Tách "rác" làm hai** bằng cách đối chiếu hộp của từng Global ID không nhãn với chú
   thích WildTrack gốc: bao nhiêu là hộp không trùng người nào (detector báo nhầm), bao
   nhiêu là người thật mà bộ gán nhãn từ chối. Con số đó quyết định phiên sau đi sửa
   detector hay đi sửa ngưỡng liên kết.
2. Nợ từ phiên 19: chạy `--publish` trên máy GPU, đọc `n_dropped`/`max_depth` để chốt trần
   hàng đợi.
3. Nợ từ phiên 15: chấm `onnx_gtbox` bằng HOTA.
