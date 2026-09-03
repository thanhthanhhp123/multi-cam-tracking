# 2026-09-03 — Khảo sát lại ut-hpc + skill `ut-hpc`

- **Mốc:** chuẩn bị M2/M3 | **Máy:** máy dev (Windows) + SSH `ut-hpc` | **Thời lượng:** ~1.5h

## Mục tiêu phiên

- Vọc kỹ `ut-hpc` (lần khảo sát 2026-08-27 chỉ ở mức "dùng được hay không").
- Đóng gói những gì học được thành một **skill** để phiên sau không phải dò lại.

## Đã làm

- Khảo sát bằng lệnh thật, không suy đoán: partition/account, cấu hình node GPU, mạng,
  lưu trữ, module, môi trường Python.
- Submit **3 job thật** để kiểm chứng (đều đã COMPLETED, hàng đợi sạch sau đó):
  - `580543` — `srun --partition=students`: pending, phải `scancel`.
  - `580545` — `sbatch --partition=main-gpu`: chạy trên `ctit091` (4× A40), thăm dò mạng/đĩa/module.
  - `580546`, `580547` — torch GPU trên `hpc-node07` (L40) bằng conda env đặt trong `$HOME`.
- Tạo skill **`.claude/skills/ut-hpc/`**:
  - `SKILL.md` — 5 điều sai là hỏng job, quy trình 5 bước, checklist trước khi submit.
  - `reference/cluster.md` — sự thật đã đo (kèm mục "chưa xác minh" tách riêng).
  - `reference/workflows.md` — quy trình M2 (YOLO) và M3 (Re-ID) + bảng chẩn đoán.
  - `scripts/hpc.sh` — `status / setup / push / submit / watch / logs / cancel / fetch / shell`.
  - `templates/` — `smoke_gpu.sbatch`, `train_yolo.sbatch`, `train_reid.sbatch` + `train_reid.py`,
    `prepare_coco_person.py`, `fetch_pretrained.py`.
- Dựng cây `~/mct/{data,jobs,runs,models,logs,.torch}` trên cụm; `hpc.sh push` + `submit` +
  `watch` đã chạy thật một vòng.
- Cập nhật CLAUDE.md §2 (bảng môi trường) và §11 (cạm bẫy): partition, mạng, container, dung lượng.

## Quyết định kỹ thuật

**1. Dùng `main-gpu` thay cho `students`.** `scontrol show partition` cho thấy `main-gpu`
(26 node GPU) và `main` để `AllowAccounts=ALL` — tài khoản sinh viên vào được, không bị bó
vào 1 node như đã tưởng hôm 2026-08-27. Đo trực tiếp: job `main-gpu` chạy sau ~40s, cùng lúc
job `students` pending với lý do *"Nodes required for job are DOWN, DRAINED or reserved for
jobs in higher priority partitions"*. Đây là khác biệt giữa "train được trong ngày" và
"chờ vô hạn định", nên chốt `main-gpu` là mặc định.

**2. Conda env trong `$HOME`, không dùng Singularity.** CLAUDE.md bản trước giả định phải
container hoá vì cụm không có Docker. Thực tế: `$HOME` là NFS, thấy được từ mọi node tính toán;
env conda đặt ở đó chạy CUDA bình thường (đã chứng minh bằng job thật). Bỏ được một tầng phức
tạp (pull `.sif` ~10G, bind mount, `--nv`) mà không mất gì. Singularity giữ làm dự phòng —
`module load singularity/3.9.5` vẫn load được trên node tính toán.

**3. Tách rõ `[HEAD]` và `[JOB]` trong mọi quy trình.** Vì node tính toán không có mạng
(xem số liệu), mọi bước chạm mạng phải nằm ở head node và **hoàn tất trước khi submit**.
Các template vì thế đều bật `ULTRALYTICS_OFFLINE=1` / `HF_HUB_OFFLINE=1` / `TORCH_HOME` cục bộ
và **kiểm tra file tồn tại rồi thoát sớm** thay vì để thư viện tự tải rồi treo tới hết `--time`.
Đây là lý do skill dành hẳn một mục "5 điều sai là hỏng job" ở đầu.

**4. Kiến thức về cụm nằm trong skill, không nằm trong CLAUDE.md.** CLAUDE.md giữ phần bất
biến của dự án; chi tiết vận hành (tên partition, dung lượng, mã lỗi SLURM) đổi theo thời gian
và chỉ cần khi thực sự đụng tới cụm — nạp theo yêu cầu qua skill thì đúng chỗ hơn. CLAUDE.md
chỉ giữ 4 dòng cạm bẫy trỏ sang.

## Số liệu đo được

Cấu hình chung: `ut-hpc` (`hpc-head1.ewi.utwente.nl`), SLURM 21.08.5, Ubuntu 22.04.5,
driver NVIDIA 595.58.03.

| Đại lượng | Giá trị |
|---|---|
| Thời gian chờ hàng đợi, `main-gpu --gres=gpu:1` | ~40s (2 lần đo, 07:00 giờ địa phương) |
| Thời gian chờ, `students --gres=gpu:1` | pending, không chạy trong ~4 phút → huỷ |
| GPU chạm được | `ctit091`: 4× A40 46068 MiB, cc 8.6 · `hpc-node07`: L40 46068 MiB |
| torch trên node tính toán | 2.13.0+cu130, `cuda_available True`, matmul 4096² ~0.87s |
| Mạng từ node tính toán (`ctit091`) | pypi / github / docker registry: **không phản hồi** |
| Mạng từ head node | cả ba: HTTP 200 / 401 (tức là tới được) |
| Home | 1.0T quota, **912G đã dùng, còn 113G** (90%) |
| Scratch cục bộ node | `/local` 2.2–5.1T trống, ghi được, **riêng từng node** |
| Python hệ thống node tính toán | 3.10.12 (khớp mục tiêu 3.10 của dự án) |
| `module load singularity/3.9.5` trên node | được, nhưng lần gọi đầu ~2 phút (NFS lạnh) |

## Vướng mắc / chưa xong

- **Chưa chạy train thật lần nào** — chưa có `~/mct/env` (bước `hpc.sh setup`, ~10G),
  chưa tải COCO/Market-1501, chưa chạy Ultralytics hay torchreid trên cụm. Template M2/M3
  mới ở mức "đúng theo hiểu biết", chưa được thực tế xác nhận.
- **Dung lượng home là rủi ro thật**: còn 113G, COCO ~20G + MSMT17 ~30G + env torch ~10G là
  vừa hết. Chưa xác định 912G đang nằm ở đâu (`du` trên NFS timeout; biết `~/miniconda3` = 40G,
  `~/.conda` = 1.6G).
- Market-1501/MSMT17 cần tải thủ công (licence) — chưa có đường tải xác nhận được.
- CLAUDE.md §2 vẫn ghi máy dev là "Mac (dev, hiện tại)" nhưng phiên này chạy trên Windows.
  Chưa sửa vì chưa rõ đây là chuyển hẳn hay tạm thời.

## Bước tiếp theo

1. `bash .claude/skills/ut-hpc/scripts/hpc.sh setup` rồi `submit smoke_gpu.sbatch` —
   xác nhận env riêng của đồ án (không mượn `fitroom`) chạm được GPU.
2. `fetch_pretrained.py` + `prepare_coco_person.py` trên head node, kiểm tra `df -h $HOME`
   trước và sau.
3. Chạy `train_yolo.sbatch` với `EPOCHS=1` để bắt lỗi template, rồi mới chạy đủ epoch.
4. Song song, không phụ thuộc cụm: `src/mct/tracklet.py` (bước tiếp theo của worklog 2026-09-01).
