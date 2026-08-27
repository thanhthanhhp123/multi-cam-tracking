# Fixture metadata

File `.jsonl` trong thư mục này **không được commit** (xem `.gitignore`). Hai loại fixture,
cách lấy khác nhau:

## 1. Fixture tổng hợp — sinh lại được

```bash
make fixture          # -> tests/fixtures/two_cam_walk.jsonl + .gt.json
```

Deterministic theo `--seed`, nên mọi máy sinh ra file giống hệt nhau. Không commit vì
845K mỗi lần sinh và chạy lại chỉ mất một giây.

Không test nào phụ thuộc vào file này — `tests/test_fixture.py` gọi thẳng `build_scenario()`
nên `make test` chạy được ngay trên bản clone sạch. File `.jsonl` chỉ dùng cho
`make replay` khi cần dữ liệu chảy qua Redis thật.

Ba tham số điều khiển độ khó, tất cả đều là cosine similarity đo được trực tiếp:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--intra-sim` | 0.80 | hai frame của cùng người, cùng camera |
| `--cross-cam-sim` | 0.75 | cùng người, hai camera khác nhau |
| `--inter-sim` | 0.65 | hai người "mặc đồ giống nhau" |

Bài toán của engine liên kết nằm ở khoảng cách giữa `cross-cam-sim` (phải khớp) và
`inter-sim` (không được khớp). Thu hẹp lại để dựng kịch bản khó:

```bash
make fixture FIXTURE=tests/fixtures/hard.jsonl   # rồi thêm --inter-sim 0.72
python -m tools.make_synthetic_fixture --inter-sim 0.72 --out tests/fixtures/hard.jsonl
```

## 2. Fixture ghi từ pipeline thật — KHÔNG sinh lại được

Ghi trên máy GPU khi pipeline DeepStream đang chạy (M3):

```bash
python -m tools.record_metadata --out tests/fixtures/hanh_lang_2cam.jsonl --duration 60
```

Đây là thứ có giá trị nhất trong cả quy trình: có nó rồi thì engine liên kết (M4) phát
triển và tinh chỉnh được hoàn toàn trên máy không GPU.

Quy ước lưu trữ:

- **Bản đầy đủ** để trong `data/fixtures/` (đã gitignore) cùng với video gốc, sao lưu
  riêng. Vài chục MB, không hợp với git.
- **Bản trích ngắn** (~30s, một kịch bản, dưới 2MB) commit vào `tests/fixtures/` bằng
  `git add -f` để test hồi quy có dữ liệu thật mà không phình repo.

Trước khi dùng một fixture thật, kiểm tra nó đúng contract:

```bash
python -m tools.record_metadata ...    # đã tự validate và cảnh báo khi ghi
```

Cảnh báo hay gặp nhất là bbox tràn khỏi khung — dấu hiệu probe quên scale toạ độ từ
`nvstreammux` về độ phân giải camera (CLAUDE.md §5).
