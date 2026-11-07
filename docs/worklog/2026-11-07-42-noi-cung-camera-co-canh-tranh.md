# 2026-11-07 (phiên 42): Nối cùng camera chỉ mất ngân sách hình học khi có ứng viên cạnh tranh

- **Mốc:** M6 (đo + sửa) | **Máy:** máy dev (Windows 10), không thuê GPU | **Thời lượng:** ~0.5h

Tiếp phiên 41, bước tiếp theo 1(a). Phiên ngắn trước buổi báo cáo GVHD.

## Mục tiêu phiên

- Thử tín hiệu (a) của phiên 41: ô nối cùng camera mất `max_cost_geometric` chỉ khi tracklet có ứng viên khác chấp nhận được.
- Có bảng số trên code hiện tại cho buổi báo cáo (bước tiếp theo 2 của phiên 41).

## Đã làm

- `association.stitch_contested_max_gap_ms` (`src/mct/affinity.py`, mặc định `null` = hành vi cũ từng bit): sau khi tính một hàng của ma trận chi phí, nếu hàng có ô KHÁC ô nối cùng camera đang dưới ngưỡng của chính nó (`max_cost` hoặc `max_cost_geometric`), thì mọi ô mà bằng chứng vị trí đến DUY NHẤT từ nối cùng camera và `Δt` > ngưỡng bị hạ về `max_cost`. Chi phí và tính khả thi không đổi, chỉ đổi cờ bằng chứng. `_pair_cost` và `_same_camera_term` trả thêm `Δt`. Khoá thêm vào `configs/mct.yaml`. +1 test (`tests/test_affinity.py`). 865 passed, 5 skipped, ruff sạch (Python 3.12, chưa chạy 3.10).
- Sweep {null, 0, 1500, 3000} + `base` trên pipeline thật (A), 12 buổi × 3 lần chạy, chạy song song 3 buổi một lúc (~8 phút): `data/lab/oracle/eval_v4/`. Script ở scratchpad (`sweep_contest.sh`, `aggregate.py`); `aggregate.py` cho ra đúng từng số của bảng phiên 41 trên `eval_v3`.

## Quyết định kỹ thuật

1. **Điều kiện "có cạnh tranh" thay cho "Δt dài".** Phiên 41 cho thấy Δt một mình không tách được "người mới vào cửa" với "cùng người hiện lại". Khác biệt có thể quan sát được: người mới vào thường có track của chính họ từ camera khác cũng qua ngưỡng, còn người bị che hiện lại thường không có ai cạnh tranh. "Chấp nhận được" = dưới ngưỡng của chính ô, để ô khả thi nhưng không bao giờ được nhận không tính là cạnh tranh.
2. **Mặc định `null` trong code (`configs/mct.yaml`); `configs/lab/lab.mct.yaml` chuyển sang `combo` + `c0`** (người dùng duyệt cuối phiên). Quy trình chọn/kiểm như phiên 41: chọn trên s2 (c0 thắng), nhóm giữ riêng cùng chiều. Đặt cả bộ `combo` (`max_cost` 0.40, `homography_weight` 0.2, `max_ground_dist_m` 3.0) chứ không chỉ thêm khoá mới, vì `base` + `c0` là cấu hình chưa đo. Kiểm: `lab.mct.yaml` mới không override, buổi s2_1 × 3 lần chạy, trùng từng số với biến thể `c0` của sweep (`data/s42/verify_s2_1/`). Từ giờ biến thể `base` của `make lab-eval` CHÍNH LÀ `c0`; muốn số `base` cũ phải override lại ba khoá.

## Số liệu đo được

Cấu hình: fixture pipeline phiên 40 (RTX A4000, YOLO11s 640 TensorRT FP16 b3, OSNet x1.0 MSDC 512, NvDCF, 3 luồng 25 fps), `configs/lab/lab.mct.yaml`, code sau bản sửa `pop_former_owner`; chấm hộp ảnh, chỉ khung chú thích, IoU 0.5. Bàn giao gộp số lần qua 3 lần chạy, khoảng Wilson 95%; HOTA/IDF1 trung bình theo buổi. `combo` = `max_cost` 0.4 + `homography_weight` 0.2 + `max_ground_dist_m` 3.0; `cN` = `combo` + `stitch_contested_max_gap_ms: N`.

| Nhóm | Biến thể | HOTA | IDF1 | Chồng lấn | Không chồng lấn | Quay lại | Tổng [95%] | đúng / tách / nhầm |
|---|---|---|---|---|---|---|---|---|
| s2 (chọn, 8 buổi) | base | 61.8 | 62.6 | 40% | 22% | 49% | 35% [29–41] | 86 / 125 / 38 |
| | combo | 68.2 | 75.5 | 67% | 45% | 47% | 55% [48–61] | 136 / 72 / 41 |
| | **c0** | **73.0** | **80.5** | 82% | 79% | 56% | **76% [70–81]** | 189 / 47 / 13 |
| | c1500 | 70.2 | 77.5 | 84% | 67% | 53% | 71% [66–77] | 178 / 48 / 23 |
| | c3000 | 68.6 | 75.7 | 81% | 59% | 51% | 67% [61–72] | 166 / 53 / 30 |
| Giữ riêng (kiểm, 4 buổi) | base | 46.1 | 42.0 | 9% | 11% | 83% | 20% [16–26] | 49 / 133 / 58 |
| | combo | 53.5 | 55.2 | 27% | 8% | 86% | 33% [28–40] | 80 / 106 / 54 |
| | **c0** | **54.9** | **58.3** | 32% | 39% | 81% | **40% [34–46]** | 96 / 100 / 44 |
| | c1500 | 54.3 | 56.4 | 35% | 36% | 89% | 43% [37–50] | 104 / 96 / 40 |
| | c3000 | 53.9 | 55.6 | 35% | 67% | 86% | 47% [41–53] | 113 / 85 / 42 |
| Cả 12 | base | 56.5 | 55.7 | 21% | 19% | 64% | 28% [24–32] | 135 / 258 / 96 |
| | combo | 63.3 | 68.8 | 42% | 36% | 64% | 44% [40–49] | 216 / 178 / 95 |
| | c0 | 67.0 | 73.1 | 51% | 68% | 67% | 58% [54–63] | 285 / 147 / 57 |

Mẫu số theo loại (cả 12 buổi, 3 lần chạy): chồng lấn 273, không chồng lấn 135, quay lại 81; nhóm giữ riêng không chồng lấn và quay lại chỉ 36 mỗi loại.

→ **Dương tính, khác `evidence_max_gap_ms` của phiên 41.** Trên nhóm giữ riêng, c0 hơn combo ở HOTA (+1.4), IDF1 (+3.1) và tổng bàn giao (+7 điểm; khoảng Wilson còn chồng nhau). Loại quay lại cùng camera giữ được (86 → 81%, n = 36, trong nhiễu), trong khi phiên 41 mất 86 → 58%. Gộp nhầm giảm 54 → 44 ở nhóm giữ riêng, 41 → 13 ở s2. Trên nhóm giữ riêng, c3000 còn cao hơn c0 về bàn giao (47%) nhưng thấp hơn về HOTA/IDF1; theo quy trình vẫn lấy c0, giá trị chọn trên s2.

**Kiểm hồi quy WildTrack trên code mới** (fixture pipeline thật YOLO11s 640 `ds_wildtrack_7cam_r640n_r{1,2,3}`, `configs/demo/wildtrack_ds.mct.yaml`, n = 3, phát lại trên máy dev; `data/s42/wt/`):

| Biến thể | Hộp ảnh HOTA (toàn khung, IoU 0.5) | IDF1 | Mặt đất, trong vùng, T = 1 m, 400 khung, không NMS | NMS 0.5 m | 40 khung test, không NMS |
|---|---|---|---|---|---|
| Số cũ (phiên 30–32) | 15.79 ± 0.47 | — | 31.1 ± 0.8 | 32.1 | — |
| base (code mới) | 15.65 ± 0.43 | 20.1 | 30.9 ± 0.7 | 31.8 ± 1.1 | 41.3 ± 1.2 |
| mix (`max_cost` 0.4 + `max_cost_geometric` 0.9) | trùng base từng số | | | | |
| mix_c0 (+ `stitch_contested_max_gap_ms` 0) | 15.71 ± 0.37 | 20.2 | 31.1 ± 0.4 | 32.1 ± 1.0 | 41.0 ± 0.2 |

→ Không hồi quy: hai bản sửa (`pop_former_owner`, nối có cạnh tranh) đều nằm trong nhiễu trên WildTrack. `mix` trùng `base` vì WildTrack đặt `ground_gap_policy: reject` và 7 camera đều chồng lấn, nên mọi ô khả thi đều có bằng chứng vị trí; tức là ngưỡng `max_cost` không bao giờ được dùng. Vì vậy WildTrack chỉ cho biết "không làm hỏng", không kiểm được lợi ích của khoá mới (lợi ích đó ở cấu hình hỗn hợp chồng lấn/không chồng lấn, đúng phần đóng góp).

## Vướng mắc / chưa xong

- Chưa chạy kịch bản B (tracker hoàn hảo) và `diag_decisions.py` trên c0 để xem gộp nhầm cùng camera còn lại bao nhiêu.
- Test chạy Python 3.12, chưa 3.10.
- Các mục còn treo của phiên 41 giữ nguyên (đồng hồ thật trên GPU, khóa API Vast.ai, `eval/gt/` + `logs/`, cặp cam02 ↔ cam03).

## Bước tiếp theo

1. Kịch bản B + `diag_decisions.py` trên c0; đưa `diag_decisions.py`, `aggregate.py` vào `eval/`.
2. Lần thuê GPU tới: chạy lại `lab-eval` + đồng hồ thật trên code này.
