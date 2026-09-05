"""Chạy TrackEval trên dữ liệu đã xuất bởi `tools/export_trackeval.py`.

    PYTHONPATH=src python eval/run_trackeval.py --root eval/trackeval --split mct

`make eval` gọi file này. Nó cố tình MỎNG: mọi kiến thức về bố cục thư mục nằm ở
`common/motformat.TrackEvalLayout`, còn phần chấm điểm là của TrackEval — viết lại HOTA hay
IDF1 bằng tay là cách chắc chắn để có một bảng điểm không ai đối chiếu được.

**TrackEval không có trên PyPI dưới dạng dùng được ngay** — phải clone từ GitHub. Bản đã
chạy được (2026-09-05, trên head node `ut-hpc` vì node tính toán không có mạng):

    git clone --depth 1 https://github.com/JonathonLuiten/TrackEval ~/TrackEval
    python3 -m venv ~/mct/venv-eval
    ~/mct/venv-eval/bin/pip install "numpy==1.23.5" "scipy==1.10.1"
    export TRACKEVAL_PATH=~/TrackEval        # hoặc dùng --trackeval-path

**Ghim `numpy==1.23.5`, không phải bản mới nhất.** TrackEval còn dùng `np.float`, thứ bị xoá
hẳn ở numpy 1.24 — chạy với 1.26 thì nổ `AttributeError` ngay lúc nạp file GT. Ghim numpy
thay vì vá bản clone, để ai tái lập cũng chỉ cần đúng những lệnh trên. Không cần
`requirements.txt` đầy đủ: chỉ numpy + scipy là chấm được MotChallenge2DBox (`pycocotools`
chỉ để cho dataset BURST, thiếu nó TrackEval in một dòng cảnh báo rồi chạy tiếp).

Venv này TÁCH khỏi `~/mct/venv-test` — venv chạy pytest phải giữ nguyên độ nhẹ.

Thiếu TrackEval thì script báo đúng câu lệnh trên rồi thoát, chứ không đổ traceback khó hiểu.

**Đọc số cho đúng:**

- `--split sct` chấm **tracker đơn camera** (mỗi camera một chuỗi, id = local track). MOTA và
  MOTP ở đây nói về detector + nvtracker, KHÔNG nói gì về `src/mct`.
- `--split mct` chấm **module liên kết** trên chuỗi ảo nối mọi camera, id = Global ID. Đây
  mới là con số của đóng góp chính. HOTA và IDF1 là hai chỉ số đáng báo cáo nhất; MOTA ở
  chế độ này chủ yếu phản ánh chất lượng detector nên đừng dùng nó để nói về liên kết.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_METRICS = ("HOTA", "CLEAR", "Identity")


def _import_trackeval(explicit: Path | None):
    """Nạp TrackEval từ `--trackeval-path`, `$TRACKEVAL_PATH`, hoặc site-packages."""
    candidates = [explicit] if explicit else []
    env = os.environ.get("TRACKEVAL_PATH")
    if env:
        candidates.append(Path(env))
    for path in candidates:
        if path and (path / "trackeval").is_dir():
            sys.path.insert(0, str(path))
            break
    try:
        import trackeval
    except ImportError:
        raise SystemExit(
            "Không tìm thấy TrackEval. Cài bằng:\n"
            "  git clone https://github.com/JonathonLuiten/TrackEval ~/TrackEval\n"
            "  pip install -r ~/TrackEval/requirements.txt\n"
            "rồi chạy lại với --trackeval-path ~/TrackEval (hoặc đặt $TRACKEVAL_PATH)."
        ) from None
    return trackeval


def build_config(
    trackeval, root: Path, *, benchmark: str, split: str, tracker: str, metrics: list[str]
):
    """Cấu hình MotChallenge2DBox khớp bố cục mà `TrackEvalLayout` dựng ra."""
    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg["PRINT_CONFIG"] = False
    eval_cfg["DISPLAY_LESS_PROGRESS"] = True
    # TrackEval vẽ đường cong HOTA bằng matplotlib SAU KHI đã tính xong, và lỗi import ở
    # đó nuốt mất toàn bộ kết quả vừa tính. Không cần hình ở đây (số nằm trong CSV) nên
    # tắt hẳn, thay vì kéo matplotlib vào venv chỉ để nó vẽ rồi vứt.
    eval_cfg["PLOT_CURVES"] = False

    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(
        {
            "GT_FOLDER": str(root / "gt"),
            "TRACKERS_FOLDER": str(root / "trackers"),
            "BENCHMARK": benchmark,
            "SPLIT_TO_EVAL": split,
            "TRACKERS_TO_EVAL": [tracker],
            "PRINT_CONFIG": False,
            # Tắt tiền xử lý của MOT Challenge: nó lọc theo lớp `distractor` và vùng
            # `zero-marked` của bộ dữ liệu gốc, thứ mà ground-truth tự thu không có. Bật
            # lên sẽ loại nhầm hộp hợp lệ.
            "DO_PREPROC": False,
        }
    )

    metric_list = []
    for name in metrics:
        cls = getattr(trackeval.metrics, name, None)
        if cls is None:
            raise SystemExit(f"TrackEval không có chỉ số {name!r}")
        metric_list.append(cls({"THRESHOLD": 0.5, "PRINT_CONFIG": False}))
    return eval_cfg, ds_cfg, metric_list


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("eval/trackeval"))
    p.add_argument("--benchmark", default="MCT")
    p.add_argument("--split", default="mct", help="'sct' (đơn camera) hoặc 'mct' (xuyên camera)")
    p.add_argument("--tracker", default="mct-engine")
    p.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    p.add_argument("--trackeval-path", type=Path, default=None)
    args = p.parse_args(argv)

    dataset_dir = args.root / "gt" / f"{args.benchmark}-{args.split}"
    if not dataset_dir.is_dir():
        raise SystemExit(
            f"{dataset_dir} không tồn tại — chạy tools/export_trackeval.py trước "
            f"(--benchmark {args.benchmark}, chế độ {args.split})."
        )

    trackeval = _import_trackeval(args.trackeval_path)
    eval_cfg, ds_cfg, metrics = build_config(
        trackeval,
        args.root,
        benchmark=args.benchmark,
        split=args.split,
        tracker=args.tracker,
        metrics=args.metrics,
    )

    evaluator = trackeval.Evaluator(eval_cfg)
    dataset = trackeval.datasets.MotChallenge2DBox(ds_cfg)
    evaluator.evaluate([dataset], metrics)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
