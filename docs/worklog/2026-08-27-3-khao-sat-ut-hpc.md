# 2026-08-27 (phiên 3) — Khảo sát ut-hpc, chốt vai trò 3 máy

- **Mốc:** trước M1 | **Máy:** Mac + SSH vào `ut-hpc` | **Thời lượng:** ~30 phút

## Mục tiêu phiên

- Kiểm tra `ut-hpc` (đã có sẵn trong `~/.ssh/config`) có dùng được để chạy DeepStream không.
- Cập nhật CLAUDE.md nếu vai trò các máy thay đổi.

## Đã làm

- SSH vào `ut-hpc` (`hpc-head1.ewi.utwente.nl`, user `s3002152`, key-based, không cần mật khẩu).
- Khảo sát: cụm SLURM thật của khoa EEMCS ĐH Twente. Ubuntu 22.04.5 LTS trên head node.
  Partition được cấp: `students`, node `hpc-node08` (4× GPU Lovelace). Không có Docker,
  có Singularity (3.7.4/3.8.0/3.9.5). Module CUDA từ 8.0 đến 12.4, TensorRT 6.0–8.6.
  Home quota 1TB, đang dùng 89%.
- Test `singularity pull docker://nvidia/cuda:...` — chạy được, pull và unpack thành công.
- **Mắc lỗi thao tác:** chạy thử `singularity exec --nv ... nvidia-smi` trực tiếp trên
  head node (không qua SLURM). Head node không có GPU nên lệnh treo, phải Ctrl-C/timeout.
  Đã dọn cache Singularity (174M) để lại trên head node, không có gì hỏng. Bài học: **mọi
  việc cần GPU trên `ut-hpc` phải qua `srun`/`sbatch`, không chạy trực tiếp trên head node**
  — đã ghi thành cạm bẫy trong CLAUDE.md §11 để không lặp lại.
- Test `srun --partition=students --gres=gpu:1 nvidia-smi` — job xếp hàng chờ (node đang bận
  do sinh viên khác dùng), không phải lỗi cấu hình. Job tự huỷ khi phiên `srun` bị ngắt,
  hàng đợi sạch sau đó (`squeue -u $USER` rỗng).
- Người dùng xác nhận **vai trò `ut-hpc`**: chỉ dùng để train/fine-tune. Khi cần chạy pipeline
  DeepStream thật, sẽ thuê GPU trên Vast.ai (đã có sẵn `Host vast-gpu` trong `~/.ssh/config`
  từ trước, IP `142.112.39.215` — nhưng instance thuê theo phiên nên có thể không còn tồn tại,
  chưa test kết nối).
- Cập nhật CLAUDE.md §2 (bảng môi trường: 2 máy → 3 máy), §9 (lộ trình: M2/M3 tách bước
  fine-tune trên `ut-hpc` khỏi bước tích hợp pipeline trên `vast-gpu`), §11 (cạm bẫy riêng
  cho từng máy: Singularity thay Docker trên `ut-hpc`, không chạy trên head node, `vast-gpu`
  thuê theo phiên nên không giả định tồn tại).

## Quyết định kỹ thuật

**Tách vai trò 3 máy thay vì 2.** Bản CLAUDE.md trước gộp "máy GPU" thành một khái niệm
duy nhất. Thực tế có `ut-hpc` (SLURM dùng chung, có GPU, không có Docker, phù hợp việc train
theo batch job) và `vast-gpu` (thuê riêng theo giờ, có Docker, phù hợp chạy pipeline cần
tương tác/realtime và cần `nvstreammux` — DeepStream runtime không có trên `ut-hpc`).

Vì sao không dùng `ut-hpc` cho cả pipeline: SLURM là mô hình batch job, không hợp với việc
"chạy pipeline realtime, xem log trực tiếp, chỉnh sửa qua lại" của M1–M3. Không có Docker
nên phải chuyển toàn bộ `docker/deepstream.Dockerfile` sang Singularity — tốn công mà
`vast-gpu` (thuê riêng, full quyền, có Docker) không cần việc đó.

Hệ quả vào lộ trình: M2 và M3 tách làm hai vế — fine-tune (YOLO trên COCO-person, OSNet trên
Market-1501/MSMT17) chạy trên `ut-hpc` bằng `sbatch`, xuất trọng số ra ONNX; tích hợp vào
pipeline + đo FPS chạy trên `vast-gpu`. Trọng số đi qua `models/` (gitignored), không qua git.

**Nguyên tắc thao tác trên `ut-hpc`:** không chạy việc nặng trên head node (dùng chung với
người khác); việc cần GPU luôn qua `srun`/`sbatch --partition=students --gres=gpu:...`;
việc chạy lâu ưu tiên `sbatch` (không đồng bộ) thay vì `srun` foreground — job xếp hàng làm
`srun` treo, dễ tưởng nhầm là lỗi.

## Số liệu đo được

—

## Vướng mắc / chưa xong

- Chưa test kết nối tới `vast-gpu` — instance có thể đã hết hạn thuê từ lần trước. Phải xác
  minh (và cập nhật `HostName`/`Port` trong `~/.ssh/config`) khi thực sự thuê để bắt đầu M1.
- Chưa viết script fine-tune nào trên `ut-hpc` — mới chỉ xác nhận môi trường dùng được.
- `docker/compose.gpu.yml` mà Makefile tham chiếu tới vẫn chưa tồn tại — sẽ tạo khi thuê
  `vast-gpu` và bắt đầu M1, không cần cho `ut-hpc`.

## Bước tiếp theo

Có hai việc độc lập, làm được song song:
1. Trên Mac: tiếp tục M4 bằng fixture tổng hợp (`src/mct/tracklet.py`) — không phụ thuộc
   máy nào ở trên.
2. Khi sẵn sàng train: viết `sbatch` script fine-tune YOLO trên `ut-hpc` (dùng module
   `nvidia/cuda-12.4` + Singularity image Ultralytics, dataset COCO-person).
