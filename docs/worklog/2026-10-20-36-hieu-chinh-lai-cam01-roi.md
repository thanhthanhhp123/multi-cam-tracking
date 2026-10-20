# 2026-10-20 (phiên 36): Chỉnh góc cam01, hiệu chỉnh lại homography cam01 bằng góc gạch + cái bát làm mốc, lọc vùng sàn theo camera (`ground_roi`)

- **Mốc:** M6 (hiện trường) + M4 (lọc vùng sàn) | **Máy:** laptop LOQ (Windows 11, không dùng GPU), 3 × Tiandy TC-C131N | **Thời lượng:** ~1.5h

## Mục tiêu phiên

- Người dùng cắm 3 camera vào laptop LOQ (bản clone mới, chưa có `.env`, ffmpeg, numpy) và giao
  "calib đi".
- Giữa phiên: merge nhánh `huy` (phiên 35, hiệu chỉnh trên laptop ZephyrusG14) vào `main`.
- Người dùng hạ cam01 cho nó bớt thấy vùng cam03, nên phải hiệu chỉnh lại cam01, rồi làm luôn
  bộ lọc vùng xa (treo từ phiên 35, quyết định 4).

## Đã làm

1. **Tìm camera.** Dây LAN không có DHCP: laptop tự lấy 169.254.249.99, còn 192.168.1.x (ở
   nhà) và dải Wi-Fi đều không có cổng 554. Gửi ONVIF WS-Discovery multicast ra cổng Ethernet,
   không cần quyền admin, thì cả 3 camera đều trả lời:
   - cam01 = `.201` (MAC `…cf:3b`), cam02 = `.85` (`…75:1f`), cam03 = `.55` (`…cf:b3`);
   - số cuối và MAC khớp phiên 35, nhưng dải đổi từ 169.254.237.x sang 169.254.249.x;
   - `.env` mới (gitignored), mật khẩu có `@` viết thành `%40`.
   - DESCRIBE Digest tự viết lần đầu báo 401: Tiandy gắn nonce theo TỪNG KẾT NỐI. Gửi xác
     thực trên cùng kết nối thì 200 OK. Cả 3 camera đã ở H.264.
2. **Môi trường laptop:**
   - ffmpeg 9.0.2 essentials (gyan.dev, sha256 khớp) ở `~/tools/ffmpeg/`;
   - venv `~/.venvs/mct-lab` (Python **3.12**, vì máy không có `uv`): numpy, scipy, PyYAML,
     dotenv, opencv-python-headless, pytest, ruff, redis, fastapi…
   - Chụp khung bằng OpenCV (backend FFmpeg), không phải chờ ffmpeg tải xong.
3. **Merge `huy` → `main`** (fast-forward 2 commit) rồi push.
4. **Kiểm cam02/cam03 còn đúng không:**
   - khung trung vị 30 ảnh / 20 s mỗi camera;
   - dò góc bàn cờ, rồi so với điểm ảnh phiên 35: cả hai camera lệch trung bình (−2, −2) px,
     **cùng một vector** → đây là thiên lệch của bộ dò, không phải camera xê dịch;
   - quy ra mét thì lệch trung vị 2–3 cm. **Giữ nguyên homography phiên 35** cho cam02/cam03.
     Phase correlation giữa các lần chụp trong phiên: ≤ 0.1 px.
5. **Hiệu chỉnh lại cam01, ba tư thế** (người dùng chỉnh camera ba lần):
   - **Tư thế 1** (hạ khoảng 7°): dò góc, lan lưới, 20 điểm. Chốt độ lệch hàng bằng phép thử
     "camera chỉ xoay": ánh xạ ảnh cũ → ảnh mới qua mặt sàn phải là K·R·K⁻¹, nên |trị riêng|
     phải bằng nhau. i0 = 0 cho độ chênh 1.006, góc 7.2°; các i0 khác ≥ 1.036 hoặc góc ≥ 16°.
     Nhưng x = 18.4 m vẫn rơi vào pixel y ≈ 62.
   - **Tư thế 2** (hạ thêm): x = 18.4 m ra khỏi khung (y ≈ −120). Phép thử phép quay hỏng ở mọi
     i0 (nghiệm "tốt nhất" là lưới lật gương). **Neo bằng cái bát**: người dùng đặt bát lên góc
     gạch (0, 0), cam02 thấy đáy bát cách điểm gốc phiên 35 dưới 2 px. Người dùng đứng ở x = 0:
     **cam01 cắt mất đầu** (chỉ thấy từ vai xuống).
   - **Tư thế 3** (ngẩng lại, giữ trọn người, vùng xa lọc bằng phần mềm): kết quả cuối.
6. **Đổi cách dò góc gạch** ở tư thế 3: dò toàn ảnh rồi ghép thì gán nhầm hàng ở xa, và vệt sáng
   của sàn bóng sinh điểm giả. Chuyển sang **dự đoán → `cornerSubPix` tại chỗ dự đoán → nhận nếu
   hội tụ < 0.12 viên và là điểm yên ngựa (giá trị lớn nhất trong lân cận 9 × 9)**, lan từng
   hàng. Kết quả 30 góc, hàng −2 … 15.
   - Loại góc dưới bát, loại hàng ≥ 11 (sát chân người đang ngồi), rồi loại góc lệch > 0.03
     viên dưới mô hình méo. Còn 16 điểm; `calibrate_homography` dùng 13.
7. **Lọc vùng sàn theo camera** (`src/mct/roi.py`, mới):
   - `GroundRoi`: hình chữ nhật trên sàn, cạnh `null` = không chặn, nửa mở `[lo, hi)`, khoá lạ
     thì báo lỗi.
   - `RoiFilter`: chiếu điểm chân qua homography; camera không khai ROI đi qua nguyên vẹn; hộp
     không chiếu được thì bỏ.
   - Khai trong `topology.yaml`: `cameras.cam01.ground_roi: {x_m: [null, 6.0]}`;
     `Topology.ground_rois`.
   - Engine lọc trong `Engine.feed`, TRƯỚC `TrackletBuilder`. Log số hộp bị bỏ. Khai ROI mà
     không có homography thì `build_engine` dừng ngay.
   - `eval/roi_filter_fixture.py --topology`: lọc fixture theo ROI từng camera bằng đúng
     `RoiFilter` đó.
   - `eval.run_lab_eval` bước 0: topology có ROI thì lọc **cả GT lẫn fixture pipeline** vào
     `<work>/roi/`, ghi `report.json`, rồi mới tới bảng GT, chấm đơn camera, engine và bàn
     giao. Bảng danh tính GT giữ nguyên.
   - `check_lab_setup`: báo OK kèm ROI, hoặc FAIL nếu camera khai ROI mà chưa hiệu chỉnh.
   - `docs/m6/README.md` mục 5, quy ước 6: chú thích cam01 dư qua mốc (chân dưới pixel
     y ≈ 270, khoảng 7.5 m); vạch cắt do bộ lọc quyết định.
   - Test: `tests/test_roi.py` (+17), +1 ở `test_run_lab_eval.py`. Test engine lab trong
     `test_check_lab_setup.py` giờ truyền `homography_dir` và kiểm có bộ lọc.
8. **Ảnh tư liệu** `data/lab/report_media/2026-10-20/` (gitignored, 84 MB, có README):
   - khung trung vị và khung cuối của mỗi lần chụp (`lan_b` … `lan_e`);
   - lưới cũ/mới vẽ đè lên ảnh, ảnh dò góc, ảnh phóng to cái bát;
   - homography phiên 35 trước khi hạ;
   - toàn bộ script (`scripts/`).
   **Ảnh có người phải làm mờ mặt.**

## Quyết định kỹ thuật

### 1. Vùng xa của cam01 lọc bằng phần mềm, không giấu bằng góc camera

- Hành lang thẳng: cam01 nhìn về phía cam02, và cam03 nằm sau cam02 trên cùng hành lang.
- Muốn mép trên khung vừa giữ đầu người ở mép xa vùng chồng lấn (cách cam01 khoảng 5.4 m),
  vừa đẩy chân người ở vùng cam03 (khoảng 21.7 m) ra ngoài, thì camera phải treo cao hơn
  1.7 × 21.7 / (21.7 − 5.4) ≈ **2.3 m**. Ở 2.7 m khe góc cũng chỉ khoảng 3°.
- Đo thật ở tư thế 2: chân người ở vùng cam03 đã ra khỏi khung, nhưng người đứng trên điểm gốc
  bị cắt đầu. Re-ID và tracking cần trọn người, nên ưu tiên giữ trọn người.
- *Loại: giữ tư thế 2.* Mất đầu người trong vùng chồng lấn.
- *Loại: chuyển cam03 đi chỗ khác.* Phía sau cam01 hành lang hết sớm; phải dựng lại cả topology.

### 2. Lọc ở engine, trước tracklet; GT đi qua cùng một hàm

- Một hệ thật biết vùng nó giám sát. Đó là tính năng của hệ, giống ROI WildTrack ở phiên 31,
  miễn là GT bị cắt **đúng cùng vạch**. Vì vậy chuỗi chấm lọc cả GT lẫn fixture bằng đúng
  `RoiFilter` mà engine dùng. Chấm đơn camera và bảng GT đọc thẳng fixture nên cần lọc trước.
- Lọc trước `TrackletBuilder`: tracklet của người đi qua mốc bị cắt đúng ở mốc, và người ở vùng
  xa không bao giờ thành tracklet, nên không tranh Global ID.
- *Loại: lọc trong pipeline DeepStream* (`nvinfer` `roi-top-offset` hoặc `nvdsanalytics`).
  Mốc theo mét cần homography, thứ chỉ có ở `src/mct`. `roi-top-offset` lại áp chung mọi
  nguồn của một `nvinfer`. Lọc trước NvDCF có thể tốt hơn một chút (phiên 31), nhưng không đáng
  thêm một nhánh cấu hình phụ thuộc phiên bản DeepStream.
- *Loại: chỉ lọc fixture lúc chấm* như phiên 31. Khi đó chế độ online, tức sản phẩm, vẫn thấy
  vùng xa.
- Khai ROI mà không có homography thì **dừng**, không chạy lặng lẽ. Khoá lạ trong `ground_roi`
  cũng **dừng**: gõ sai thành "không chặn gì" là đúng loại lỗi không triệu chứng của §11.

### 3. Mốc 6 m

- Homography cam01 ở tư thế 3 còn chính xác tới khoảng 5.5 m (mọi góc gạch lệch ≤ 11 cm). Xa
  hơn thì méo ống kính làm sai tới 40 cm ở 10 m.
- Từ 6 m đến 18.4 m là vùng mù cố ý, giữ đúng `distance_m` = 12.4 m mà phiên 35 đã tính.
- Chỉ lọc theo x; theo y hành lang đã tự giới hạn bằng tường.

### 4. Neo lưới vào hệ toạ độ chung bằng một đồ vật thấy được ở cả hai camera

- Phép thử "chỉ xoay" chỉ đúng khi tâm quang học không dời. Ở tư thế 2 nó hỏng: chỉ có 2 đường
  ron dọc, quá yếu để tách xoay khỏi dời; ước lượng tiêu cự lệch 2.4 lần giữa hai lần chụp.
- Cái bát đặt trên góc gạch: cam02 (homography không đổi) cho ra toạ độ mét, cam01 cho ra chỉ
  số lưới. Chỉ cần đúng tới nửa viên (34 cm), và cách này không phụ thuộc camera xoay hay dời.
  Từ nay đây là quy trình chuẩn khi chạm vào camera.

### 5. Méo ống kính: đo, nhưng chưa đưa vào pipeline

- Mô hình division một tham số, tâm ở tâm ảnh. k1 ≈ −0.072 ở tư thế 2 (35 điểm, hai tư thế
  chung một k1) và −0.069 ở tư thế 3, ước lượng độc lập. Méo là tính chất của ống kính, và hai
  số trùng nhau.
- Bỏ qua méo thì sai số lớn nhất của homography thuần khoảng 7 cm trong vùng dùng (x < 5.5 m),
  nhỏ hơn nhiều so với sai số điểm chân người (hàng chục cm). Vì vậy chưa thêm bước khử méo
  vào `HomographyMapper`.
- Ghi cho chương 4: sàn càng chiếm nhiều chiều cao ảnh thì méo càng đáng kể.

## Số liệu đo được

Cấu hình: 3 × Tiandy TC-C131N, H.264 1920×1080 25 fps, RTSP/TCP qua LAN link-local, laptop LOQ.
Homography qua `tools.calibrate_homography` (cắt 10% điểm sai nhất, 2 vòng), bước lưới 0.6815 m.

| | ảnh dùng | điểm dùng | RMSE | p95 | ghi chú |
|---|---|---|---|---|---|
| cam01 tư thế 1 | `2026-10-20/` | 16/20 | 0.7 cm | 1.2 cm | neo bằng phép thử phép quay |
| cam01 tư thế 2 | `2026-10-20b/` | 13/15 | 1.0 cm | 1.8 cm | neo bằng cái bát |
| **cam01 tư thế 3 (cuối)** | `2026-10-20e/` | **13/16** | 1.3 cm | **2.1 cm** | neo bằng cái bát |
| cam02 (phiên 35, giữ) | — | 10/12 | 0.7 cm | 1.0 cm | hôm nay lệch trung vị 2 cm |
| cam03 (phiên 35, giữ) | — | 12/14 | 1.1 cm | 1.7 cm | hôm nay lệch trung vị 3 cm |

Kiểm chéo giữa hai camera:
- **Cái bát, tư thế 3:** cam01 cho (0.028, −0.004) m, cam02 cho (0.005, 0.003) m → **lệch 2.4 cm**.
  Ở tư thế 2: 3 cm.
- **Người đứng ở x = 0, tư thế 2:** điểm chân lệch **7 cm** giữa hai camera.
- **Người đứng ở x ≈ 3.1 m, tư thế 3:** lệch 0.70 m. Nguyên nhân là **chân bị cắt ở mép dưới
  cam02**: mép dưới cam02 ứng với x ≈ 2.5 m, ngoài vùng có điểm hiệu chỉnh của cam02. Số đo
  của cam01 khớp với việc đếm gạch.
  → **Vùng chồng lấn thấy trọn người ở cả hai camera chỉ là x ∈ [−1, 2] m**, khoảng 4–5 hàng
  gạch quanh điểm gốc.
- **Độ chính xác cam01 theo khoảng cách** (mọi góc dò được, chiếu qua homography cuối):
  - x từ −1.4 tới 5.5 m: lệch ≤ 11 cm, phần lớn ≤ 2 cm;
  - 7.5 m: −32 cm; 10.2 m: −43 cm.
- Test: **824 passed, 12 skipped** (venv `mct-lab`, **Python 3.12**, không phải 3.10). Phiên 34
  có 810 test. ruff check/format sạch.

## Vướng mắc / chưa xong

- **Chưa chạy bộ test bằng Python 3.10.** Laptop này không có `uv`; chạy lại trên máy dev bằng
  `~/.venvs/mct-test`. `roi.py` không dùng gì từ 3.11 trở lên.
- **Chưa kiểm điểm chân bằng người thật ở tư thế 3, trong vùng cả hai cùng thấy chân** (x từ 0
  tới 2 m). Kiểm cái bát lệch 2.4 cm; kiểm bằng người ở tư thế 2 lệch 7 cm.
- IP link-local có thể đổi khi camera khởi động lại (hôm nay đã đổi từ .237 sang .249). Nên đặt
  IP tĩnh trên trang web của camera trước buổi quay, hoặc dò lại bằng WS-Discovery.
- Script dò góc, đo méo và WS-Discovery chỉ nằm trong `report_media/…/scripts`, chưa vào repo.
  Nếu còn phải hiệu chỉnh lại nhiều lần thì nên gom thành một công cụ.
- `.env` (có mật khẩu camera) và venv nằm trên laptop LOQ. Xoá sau khi xong M6 nếu là máy mượn.
- Transit `min_ms`/`max_ms` vẫn là số giữ chỗ (topology `status: template`), cần buổi `calib`.
- Vẫn treo: kết quả buổi gặp GVHD.

## Bước tiếp theo

1. **Không chạm vào camera nữa.** Chạm vào thì đặt cái bát lên góc gạch gốc rồi hiệu chỉnh lại
   theo mục 5–6 (script trong `report_media/2026-10-20/scripts`).
2. Quay `calib` rồi `s1`–`s4` theo `docs/m6/checklist-quay-hanh-lang.md`. Kịch bản chồng lấn đi
   trong x ∈ [−1, 2] m.
3. Chép `data/lab/raw/` và `data/lab/report_media/` về máy dev. Chạy bộ test bằng venv 3.10.
