# ut-hpc — sự thật đã xác minh

Khảo sát **2026-09-03** (lần đầu 2026-08-27). Mọi số dưới đây là **đo được**, không suy đoán.
Phần "chưa xác minh" ghi rõ ở cuối — đừng trình bày nó như sự thật.

## Truy cập

| | |
|---|---|
| Host SSH | `ut-hpc` → `hpc-head1.ewi.utwente.nl`, user `s3002152`, key-based, không mật khẩu |
| OS head node | Ubuntu 22.04.5 LTS |
| SLURM | 21.08.5 |
| Account SLURM | `students` (một association duy nhất, không giới hạn partition) |
| Group | `students b-tcs` |

Head node **không có GPU**. `nvidia-smi` ở đó sẽ treo — sai lầm đã mắc phiên 2026-08-27.

## Partition — điểm quan trọng nhất

`scontrol show partition` cho thấy hai partition mở cho mọi account:

| Partition | Nodes | AllowAccounts | Ghi chú |
|---|---|---|---|
| `main-gpu` | 26 node GPU (`ctit084-094`, `hpc-node01-12,14,16-18`) | **ALL** | **Dùng cái này** |
| `main` | 32 node (mặc định) | ALL | Gồm cả node CPU |
| `main-cpu` | 6 node CPU | ALL | Việc không cần GPU |
| `students` | **1 node** (`hpc-node08`) | admin, students | Hay pending, tránh |
| còn lại (`dmb`, `mia`, `am`, `bss`, …) | | account riêng của nhóm nghiên cứu | Không vào được |

Mọi partition đều `PriorityTier=1`, `MaxTime=UNLIMITED`.

**Đo hôm 2026-09-03:** job `--partition=main-gpu --gres=gpu:1` chạy sau ~40s.
Cùng lúc đó job `--partition=students` pending với lý do
*"Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions"*.
→ worklog 2026-08-27 ghi "partition được cấp: students" là **chưa đủ**; `main-gpu` mới là
đường nhanh.

## GPU

Node nào cũng được, tuỳ SLURM xếp. Đã chạm thực tế:

| Node | GPU | VRAM | Driver | Compute cap |
|---|---|---|---|---|
| `ctit091` | 4× NVIDIA A40 | 46 GB | 595.58.03 | 8.6 (Ampere) |
| `hpc-node07` | NVIDIA L40 | 47.7 GB | — | (Lovelace) |
| `hpc-node08` | 4× Lovelace (L40) | — | — | partition `students` |

Còn có `blackwell`, `hopper`, `turing` ở các partition khác (không vào được).
VRAM 46–48 GB → rộng rãi cho fine-tune YOLO/OSNet; không cần bận tâm batch size như trên
GPU 8GB của `vast-gpu`.

Muốn GPU cụ thể: `--gres=gpu:ampere:1` hoặc `--gres=gpu:lovelace:1`. Không cần thiết
cho đồ án này — cứ `--gres=gpu:1`.

## Mạng — cạm bẫy số 1

| Nơi | pypi.org | github.com | registry-1.docker.io |
|---|---|---|---|
| Head node | HTTP 200 | HTTP 200 | HTTP 401 (tức là *tới được*) |
| Node tính toán (`ctit091`) | **không phản hồi** | **không phản hồi** | **không phản hồi** |

Không có biến `*_proxy` nào được đặt. → Node tính toán bị cách ly mạng.

Hệ quả bắt buộc:
- Tạo/cập nhật env Python: **head node**.
- Tải dataset, pretrained weight: **head node**.
- `singularity pull docker://…`: **head node** (đã test được phiên 2026-08-27).
- Job chỉ đọc từ `$HOME`. Bật `ULTRALYTICS_OFFLINE=1`, `HF_HUB_OFFLINE=1`,
  `TORCH_HOME=$HOME/mct/.torch` để thư viện không lén gọi mạng.

## Lưu trữ

| Đường dẫn | Kích thước | Ghi được? | Chia sẻ giữa node? |
|---|---|---|---|
| `$HOME` (`/home/s3002152`, NFS storage4) | 1.0T, **còn 113G** (90% đầy) | có | **có** |
| `/local` | 2.2–5.1T trống, world-writable | có | **không** — riêng từng node |
| `/local_home` | 315G | không | không |
| `/projects`, `/deepstore`, `/datasets/*` | hàng chục TB | **không** | có (chỉ đọc, và chỉ dir của nhóm khác) |

Chỉ `$HOME` vừa ghi được vừa thấy từ mọi node → dataset và env phải nằm ở đó.
`/local` chỉ làm scratch trong phạm vi một job (copy vào đầu job, xoá cuối job).

Đang chiếm chỗ trong home: `~/miniconda3` 40G, `~/.conda` 1.6G, `~/LeeHoang_` (lớn, chưa đo xong).
`df -h $HOME` là nguồn số chính xác, `du` trên NFS rất chậm (hay timeout).

## Môi trường phần mềm

- `module` cần shell login: `ssh ut-hpc 'bash -lc "module avail"'` — trong shell
  non-interactive lệnh `module` **không tồn tại**. Trong `.sbatch` thì
  `source /etc/profile.d/modules.sh` trước.
- Modules có ích: `nvidia/cuda-12.4`, `nvidia/cuda-11.x_tensorrt-8.6`, `python/3.10.7`,
  `singularity/3.7.4|3.8.0|3.9.5`, `anaconda3/*`, `miniconda3/*`, `cmake/3.31.11`.
  Không có CUDA 12.5+ dạng module; driver 595 mới nên wheel `cu121/cu124/cu126/cu130` đều chạy.
- `module load singularity/3.9.5` đặt sẵn `SINGULARITY_TMPDIR=/local`,
  `SINGULARITY_BIND=/deepstore,/software`. Lần gọi đầu chậm (~2 phút, NFS lạnh).
- **Không có Docker.**
- Python hệ thống trên node tính toán: **3.10.12** (khớp mục tiêu 3.10 của dự án).
- Có sẵn: `git`, `rsync`, `curl`, `tclsh`. Không có `nvcc` nếu chưa load module.

## Env Python — cách đã kiểm chứng

Conda env đặt trong `$HOME` (NFS) chạy được trên node tính toán, có CUDA, **không cần container**.
Bằng chứng: env sẵn có `~/miniconda3/envs/fitroom` (Python 3.10.20, torch 2.13.0+cu130)
submit qua `sbatch --partition=main-gpu --gres=gpu:1` → chạy trên `hpc-node07`:

```
torch 2.13.0+cu130 cuda_available True
device NVIDIA L40
matmul4096 ok in 0.87s
mem_total_GB 47.67
```

→ Đường chuẩn cho đồ án: **conda env trong home**, tạo trên head node, dùng trong sbatch.
Singularity giữ làm phương án dự phòng (CLAUDE.md §11 viết trước khi biết điều này).

Env sẵn có của người dùng (đừng đụng vào, không phải của đồ án này):
`fitroom` (py3.10, torch 2.13+cu130), `captcha` (py3.11, torch 2.7.1+cu118),
`fitroom-mask` (py3.9, torch 1.10.1), `medner-topic2` (py3.11).

## Chưa xác minh — đừng nói chắc

- Chưa chạy thật một lệnh train Ultralytics hay torchreid nào trên cụm.
- Chưa tải dataset nào (COCO-person / Market-1501 / MSMT17) về cụm.
- Chưa xuất ONNX trên cụm.
- Chưa biết 912G trong home là của thư mục nào (`du` timeout trên NFS).
- Chưa test `singularity exec --nv` chạy được payload GPU (mới chỉ `--version`).
