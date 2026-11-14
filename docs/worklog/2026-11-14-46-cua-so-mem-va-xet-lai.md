# 2026-11-14 — Cửa sổ transit mềm + xét lại quyết định đầu (rv50), chấm LOSO và nhóm giữ riêng

- **Mốc:** M6 (cải tiến liên kết) | **Máy:** Mac (CPU, phát lại fixture) | **Thời lượng:** ~6h

## Mục tiêu phiên

- Tăng độ chính xác liên kết, theo các hướng trong tài liệu AI City Challenge: lọc embedding theo chất lượng crop (OCMCTrack, Specker 2021), xét lại quyết định theo trạng thái (Xie et al. CVPRW 2024, Li 2019).
- Giữ quy trình phiên 45:
  - khoá mới mặc định `null`;
  - LOSO trên 8 buổi s2, quy tắc chọn phiên 43;
  - không học gì từ embedding, không fine-tune, không sửa GT;
  - nhóm giữ riêng mở một lần ở cuối, sau khi người dùng duyệt.

## Đã làm

0. **Chẩn đoán quyết định gán đầu** trên c0 + foot, 8 buổi s2 × r1–r3 (`data/lab/s46/step0.md`).
   - Kết quả: đúng-gộp 231, đúng-mới 91, gộp nhầm 51, tách 35.
   - Gộp nhầm theo loại được chọn: cùng camera 31, không chồng lấn 14, chồng lấn 6.
   - 12/86 lỗi có người đúng bị loại vì **"quá lâu"** (26–37 s > `max_ms` 19.4 s, cặp cam01↔cam03). 4 lỗi khác là "quá nhanh" (do dây chuyền lỗi trước đó).
   - Trong gộp nhầm cùng camera, ứng viên sai thường có ngoại hình GẦN hơn ứng viên đúng. → Bỏ ý định siết nối cùng camera bằng ngoại hình (bước 2 ban đầu), thay bằng cửa sổ mềm.
1. **`tracklet.embed_max_overlap` / `embed_edge_margin_px`** (`src/mct/tracklet.py`, commit `23e9a44`). Embedding của crop bị che hơn ngưỡng, hoặc chạm mép ảnh, xếp sau các crop sạch khi lấy top-k. Không còn crop sạch thì dùng lại tất cả. 4 test.
2. **`association.transit_overdue_max_cost`** (`src/mct/affinity.py`, commit `326d3db`).
   - Ô đến QUÁ MUỘN so với `max_ms` (cặp có transition, không chồng lấn, khác camera) không bị loại nữa; vẫn ghép được nếu chi phí ngoại hình < ngưỡng.
   - `min_ms` vẫn là giới hạn cứng.
   - 1 test.
3. **`eval.loso_eval` chấm nghiêm khi bật `revise_after_frames`** (commit `773dccf`): chạy engine, đo thời gian tới Global ID, vá SQLite bằng `eval.strict_revise_score` rồi chấm lại. ID tạm giữ các khung đã hiển thị trước lần xét lại.
4. **LOSO ba bước** (base → +soft → +soft+rv), lượt chọn cuối trên cả 8 buổi, kiểm WildTrack.
5. **`configs/lab/lab.mct.yaml`: `transit_overdue_max_cost: 0.35`, `revise_after_frames: 50`** (người dùng duyệt, commit `4c931b4`). Sau đó chạy nhóm giữ riêng một lần.
6. Bộ sơ đồ tiếng Anh cho báo cáo/slide: `docs/figures/` gồm 6 hình (kiến trúc, pipeline, thuật toán, bố trí camera, quy trình dữ liệu, giao thức đánh giá). Mỗi hình có `.dot` + `.svg` + `.png`, vẽ lại bằng `build.sh`.

**Sự cố, ghi để tránh lặp lại:**
- Lần chạy LOSO đầu chạy song song với lúc đang sửa `tracklet.py`, nên chết vì `NameError`. Lần chạy lại dùng cache, mà một phần cache sinh từ code đang sửa dở, nên ra HOTA 68.9 sai.
- Nguyên nhân: khoá cache của `loso_eval` gồm cấu hình và nội dung file học được, KHÔNG gồm code.
- Sửa: xoá `loso/runs`, chạy lại từ đầu. Kết quả ra đúng số phiên 45.
- Từ đó LOSO chỉ chạy từ một git worktree đóng băng (`run46`, detached), không chạy từ cây đang sửa.
- `--keep-runs` tốn 23 GB đĩa; đã xoá.

## Quyết định kỹ thuật

1. **Bộ lọc embedding: âm tính, giữ `null`.**
   - Ngưỡng che 0.2 đánh dấu 12% số hộp nhưng gần như không đổi kết quả: crop bị che vốn đã có confidence thấp nên đã nằm ngoài top-8.
   - Lề mép 10 px tệ hơn rõ: 17% số hộp bị đánh dấu, mà crop sát camera lại là crop tốt nhất.
   - LOSO trùng từng số với base.
2. **Cửa sổ mềm 0.35: giữ.** Tăng nhỏ nhưng cùng chiều ở mọi ngưỡng thử (t20–t35), nhầm 18 → 15. Ngưỡng 0.35 chặt hơn `max_cost` 0.40: ghép muộn phải giống hơn bình thường.
3. **`revise_after_frames: 50`: giữ.**
   - Các fold chia đều 4/4 giữa rv25 và rv50.
   - Lượt chọn trên 8 buổi ra rv50: HOTA 79.72 so với 79.71, nhầm 13 so với 17.
   - rv12 tệ hơn (nhầm 22): xét lại quá sớm thì bằng chứng chưa đủ.
   - Phiên 44 chọn rv25. Hai giá trị sát nhau, nên đây không phải mâu thuẫn.
4. **Điều kiện chấp nhận rv** (đặt trước khi chạy): nhầm trong LOSO không được tăng. Kết quả 15 → 15, đạt.
5. **Vì sao chọn trên s2 và kiểm trên s1/s3/s4** (giải thích cho người dùng, ghi để đưa vào chương 5):
   - Chỉ s2 có đủ buổi cùng kịch bản (8) để làm LOSO.
   - Nhóm giữ riêng khác KỊCH BẢN (chồng lấn s1, đông s3, áo cùng màu s4), không chỉ khác buổi.
   - Cái giá: số LOSO đại diện cho cảnh thưa 2 người.

## Số liệu đo được

Cấu hình chung:
- fixture pipeline thật `data/lab/gpu_s42` (YOLO11s 640 + NvDCF + OSNet x1.0 ReID, 3 camera 1920×1080, 25 fps, 3 lần chạy r1–r3 mỗi buổi);
- phát lại engine trên CPU (Mac), `configs/lab/lab.mct.yaml` (c0 + `foot_from_height`);
- TrackEval hộp ảnh, chỉ khung có GT, IoU 0.5.

**LOSO 8 buổi s2** (mô hình chân khớp lại mỗi fold; HOTA/AssA/IDF1 là trung bình 8 buổi test; bàn giao gộp qua các fold; các dòng có rv được **chấm nghiêm**):

| bước | HOTA | AssA | IDF1 | chồng lấn | không chồng lấn | quay lại | tổng [Wilson 95%] | đúng/tách/nhầm | tới Global ID p50 / p95 (ms) | chọn theo fold |
|---|---|---|---|---|---|---|---|---|---|---|
| base (phiên 45) | 71.5 | 62.0 | 78.8 | 91% (96/105) | 65% (64/99) | 53% (24/45) | 74% [68–79] | 184/47/18 | 439 / 679 | — |
| +emb (âm tính) | 71.5 | 62.0 | 78.8 | 91% | 65% | 53% | 74% [68–79] | 184/47/18 | 439 / 679 | o20×4, o60×3, off×1 |
| +soft | 71.8 | 62.6 | 79.2 | 92% (97/105) | 68% (67/99) | 53% (24/45) | 76% [70–80] | 188/46/15 | 439 / 679 | t35×6, t30×1, t20×1 |
| +soft+rv | **79.5** | **75.8** | **90.4** | 94% (99/105) | 80% (79/99) | 58% (26/45) | **82% [77–86]** | 204/30/15 | 439 / 679 | t35rv25×4, t35rv50×4 |

- Chấm nghiêm có tác dụng thật: 214/228 lần chạy rv có ID bị vá, và điểm sau vá thấp hơn điểm chấm lỏng. Ví dụ s2_1 rv25: HOTA 84.4 khi chấm lỏng, 84.0 khi chấm nghiêm.
- Cả 8 buổi với mô hình chân khớp trên cả 8 (cấu hình cố định):

| | t20 | t25 | t30 | t35 | t35rv12 | t35rv25 | t35rv50 | t30rv25 |
|---|---|---|---|---|---|---|---|---|
| HOTA | 72.11 | 72.11 | 72.12 | 72.12 | 75.20 | 79.71 | 79.72 | 79.71 |
| đúng/tách/nhầm | 190/44/15 | 190/44/15 | 187/46/16 | 188/45/16 | 188/39/22 | 202/30/17 | 207/29/13 | 202/30/17 |

**WildTrack, kiểm hồi quy:**
- Fixture `ds_wildtrack_7cam_r640n_r{1,2,3}`, `configs/demo/wildtrack_ds.mct.yaml`, mọi khoá mới để `null`.
- `sqlite3 .dump` ở `1e99d90` (đầu phiên 45) và `773dccf` **trùng md5 cả ba lần chạy** (385 / 429 / 403 Global ID). Không hồi quy.

**Nhóm giữ riêng (s1_1, s3_1, s3_2, s4_1):**
- Chạy MỘT lần, 2026-11-14, sau khi người dùng duyệt; không chỉnh gì sau khi thấy số. Đây là lần nhìn thứ 6, nên trình bày như "kiểm lặp".
- Chấm nghiêm (số ID bị vá: 7 / 35 / 14 / 16).
- Script và kết quả: `data/lab/s46/heldout/` (`heldout46.py`, `scores.json`; gitignored).

| cấu hình | HOTA | AssA | IDF1 | chồng lấn | không chồng lấn | quay lại | tổng [Wilson 95%] | đúng/tách/nhầm | tới Global ID p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| c0 + foot (phiên 45) | 53.0 | 36.0 | 56.0 | 42% (71/168) | 53% (19/36) | 72% (26/36) | 48% [42–55] | 116/70/54 | 434 / 679 |
| **+ t35 + rv50** | **61.2** | **47.6** | **67.2** | 54% (91/168) | **33% (12/36)** | 92% (33/36) | 57% [50–63] | 136/47/57 | 436 / 680 |

Theo buổi (HOTA, đúng/tách/nhầm): s1_1 74.4, 25/5/3; s3_1 54.4, 50/20/23; s3_2 54.0, 30/9/24; s4_1 62.2, 31/13/7.

→ Ở nhóm giữ riêng:
- **Tăng:** HOTA +8.2, IDF1 +11.2, tổng bàn giao 48 → 57% (khoảng Wilson chồng nhau), chồng lấn 42 → 54%, quay lại 72 → 92%.
- **Giảm:** không chồng lấn 53 → 33% (19 → 12 trên 36, n nhỏ), nhầm 54 → 57.
- Không chồng lấn đi ngược chiều với LOSO (65 → 80%). Chưa chẩn đoán vì sao. Theo quy trình thì không chỉnh gì nữa.

## Vướng mắc / chưa xong

- **Không chồng lấn giảm ở nhóm giữ riêng (19 → 12/36).** Nghi do cửa sổ mềm hoặc xét lại ở cảnh đông/áo cùng màu: ghép muộn dựa vào ngoại hình, mà s4 áo cùng màu, s3 nhiều người. Muốn chẩn đoán thì chỉ được làm trên s2 hoặc một tập mới; đào vào nhóm giữ riêng là phá quy trình.
- Nhóm giữ riêng đã bị nhìn 6 lần. Một cải tiến tiếp theo nên có dữ liệu kiểm mới (quay thêm), không nên dùng lại nhóm này.
- Chưa đo đồng hồ thật (vast-gpu) với rv50. Thời gian tới Global ID theo thời gian dữ liệu không đổi, vì ID tạm vẫn cấp ngay.
- Deck/báo cáo cần cập nhật số theo cấu hình mới.
- Nhánh `phien-46-chat-luong-lien-ket` chưa push.

## Bước tiếp theo

1. Chẩn đoán vì sao không chồng lấn giảm, trên s2 bằng `eval.diagnose_*`: tách riêng tác dụng của t35 và rv50 trên các lần chuyển không chồng lấn. Không mở lại nhóm giữ riêng.
2. Cập nhật deck/chương 6 theo bảng trên (LOSO + nhóm giữ riêng, ghi rõ chấm nghiêm, lần nhìn thứ 6).
3. Đưa `docs/figures/` vào báo cáo.
