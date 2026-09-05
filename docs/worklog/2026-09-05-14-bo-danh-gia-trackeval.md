# 2026-09-05 (phiên 14) — Bộ đánh giá chuẩn: MOT/TrackEval + ground-truth từ CVAT

- **Mốc:** M6 (hạ tầng đánh giá) | **Máy:** máy dev (soạn) + `ut-hpc` (chạy test/lint) | **Thời lượng:** ~1h, **không tốn GPU**

## Mục tiêu phiên

- Đóng phần việc còn dở đang nằm ngoài git: bộ xuất TrackEval + bộ đọc CVAT (6 file, chưa
  commit, chưa chạy test lần nào).
- Viết test cho `tools/cvat_to_mot.py` — hai file mới kia có test, file này thì không.
- Chạy `pytest` + `ruff` trên `ut-hpc` (Python 3.10.12) rồi commit.

## Đã làm

- **`tests/test_cvat_to_mot.py`** (mới, 19 test / 21 ca) — canh ba nhóm tính chất: khung
  CVAT đếm-từ-0 → MOT đếm-từ-1, danh tính nối xuyên camera bằng thuộc tính `person_id`, và
  **mọi chú thích hỏng phải nổ chứ không đi tiếp im lặng**. Có một test hợp đồng chạy thẳng
  đầu ra của `cvat_to_mot` qua `export_trackeval.load_gt_table` — nếu hai bên lệch định dạng
  thì quy trình M6 đứt ở giữa, và đó là kiểu lỗi không có triệu chứng.
- **Sửa hai lỗ im lặng lộ ra khi viết test** (chi tiết ở phần quyết định):
  `MotRow.visibility` không sống sót qua `write_gt`; hộp CVAT thiếu toạ độ thành `NaN`.
- Dọn lint của bốn file chưa commit: `RUF100`, `E501`, `I001`, `RUF043` + `ruff format`.
- Chạy `pytest` và `ruff` trên `ut-hpc`: **430 passed, 8 skipped**, ruff sạch, 70 file đã
  đúng format.
- Smoke-test đường lỗi của `eval/run_trackeval.py`: thư mục chưa xuất thì nó dừng ngay với
  câu "chạy `tools/export_trackeval.py` trước", **không** đổ traceback của TrackEval.
- Commit cả 7 file (bộ đánh giá vốn nằm ngoài git từ phiên trước).

## Quyết định kỹ thuật

**1. `visibility` là thuộc tính của HỘP, không phải tham số lúc ghi file.** `MotRow.as_gt_line`
nhận `visibility` qua keyword, nhưng `write_gt(rows)` chỉ nhận danh sách hộp nên **không có
đường nào truyền giá trị đó vào** — mọi dòng ground-truth ra `1.00`. Trong khi đó docstring
của `cvat_to_mot` ghi rõ quy ước 4: hộp `occluded="1"` phải ghi `visibility=0.5`. Tức quy ước
đã công bố cho người gán nhãn nhưng chưa bao giờ được cài đặt. Sửa: đưa `visibility` thành
trường của `MotRow` (mặc định 1.0), `cvat_to_mot` đặt 0.5 khi `occluded`. Đây đúng loại lỗi
đáng sợ nhất của cả nhóm file này — TrackEval vẫn chấm, bảng điểm vẫn ra số, không có gì báo.

**2. Hộp thiếu toạ độ phải nổ, không được thành `NaN`.** `parse_cvat_video` dùng
`box.get("xtl", "nan")`, nên chú thích thiếu thuộc tính đi thẳng qua `float()` thành `NaN`,
rồi ghi ra `gt.txt` dưới dạng chuỗi `nan` — mà `except (TypeError, ValueError)` ngay bên dưới
lại mang thông báo "hộp thiếu toạ độ", tức code *tưởng* mình đã canh trường hợp đó. Đổi sang
`box.attrib[...]` + bắt thêm `KeyError`. Cùng lý do với (1): file điểm vẫn sinh ra được.

**3. Test đặt ở ranh giới giữa hai công cụ, không chỉ trong từng công cụ.**
`cvat_to_mot` (sinh bảng danh tính) và `export_trackeval` (đọc bảng đó) là hai file khác
nhau, cùng phải khớp một định dạng `.gt.json` đã có sẵn hai bản cài đặt khác
(`wildtrack_to_fixture.py`, `ds_wildtrack_gt.py`). Bốn bản cài đặt của một định dạng là công
thức chắc chắn để lệch. Nên test gọi thẳng hàm đọc thật của bên kia thay vì so với hằng số
chép tay.

**4. Không tự cài đặt lại HOTA/IDF1.** `run_trackeval.py` cố tình mỏng: mọi hiểu biết về bố
cục thư mục nằm ở `common/motformat.TrackEvalLayout`, phần chấm điểm để TrackEval lo. Chỉ số
tự viết là chỉ số không ai đối chiếu được với công bố nào — mà chương 6 cần đúng thứ đối
chiếu được. `DO_PREPROC=False` vì tiền xử lý của MOT Challenge lọc theo lớp `distractor` và
vùng zero-marked của bộ gốc, thứ ground-truth tự thu không có.

## Số liệu đo được

**Cấu hình:** `ut-hpc` head node, `~/mct/venv-test` (Python 3.10.12, không torch).

| | trước phiên | sau phiên |
|---|---|---|
| test | 363 passed, 8 skipped (phiên 13) | **430 passed, 8 skipped** (11.3 s lần đầu, 3.1 s lần sau) |
| ruff | 4 lỗi + 2 file sai format (nhóm file chưa commit) | sạch, 70 file đúng format |

Chưa có MOTA/IDF1/HOTA — TrackEval chưa được cài, và chưa xuất dữ liệu lần nào.

## Vướng mắc / chưa xong

- **Cả bộ đánh giá chưa từng chạy trên dữ liệu thật.** Test dùng dữ liệu dựng tay; đường
  `export_trackeval → run_trackeval → bảng điểm` chưa đi trọn một lần. Cần: cài TrackEval
  trên head node `ut-hpc` (có Internet, chỉ cần numpy/scipy), xuất từ fixture
  `ds_wildtrack_7cam` + SQLite store, rồi chạy chấm. Đây là lần đầu đồ án có chỉ số so được
  với công bố ngoài — tới giờ mới chỉ có F1 theo cặp tracklet, một chỉ số tự chế.
- `cvat_to_mot.py` chưa gặp file CVAT thật lần nào (chưa có dữ liệu tự thu). Test dựng XML
  tối thiểu theo tài liệu định dạng, nên vẫn còn giả định **chưa xác minh**: CVAT xuất
  thuộc tính đúng chỗ như mô tả. Kiểm được ngay khi có task CVAT đầu tiên.
- Quy ước chú thích (mỗi người một track, `person_id` giống nhau ở mọi camera) mới nằm trong
  docstring. Phải phổ biến cho người gán nhãn **trước khi** họ bắt đầu, không phải sau.
- Hướng đi chính của M4 từ phiên 13 vẫn treo nguyên: làm mượt điểm chân, siết
  `max_ground_dist_m`, ghép tracklet trong cùng camera.

## Bước tiếp theo

1. Cài TrackEval trên head node `ut-hpc`, chạy trọn `export_trackeval --mode sct` rồi
   `--mode mct` trên `ds_wildtrack_7cam` → **bộ MOTA/IDF1/HOTA đầu tiên** cho chương 6.
2. Quay lại việc M4 của phiên 13: làm mượt điểm chân trong `src/mct/tracklet.py`, đo lại
   `d_ground` bằng `diagnose_tracklets`.
3. Ghép tracklet trong cùng camera.
