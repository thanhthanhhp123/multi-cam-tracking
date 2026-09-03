"""Dựng pipeline DeepStream: nguồn → nvstreammux → nvinfer → nvtracker → probe.

CHỈ chạy được trên máy có GPU + DeepStream runtime (CLAUDE.md §2). Cùng với
`probes.py`, đây là chỗ duy nhất được import `gi`/`pyds`.

Sơ đồ (CLAUDE.md §3):

    nvurisrcbin ─┐
    nvurisrcbin ─┼→ nvstreammux → nvinfer(PGIE YOLO) → nvtracker → pad probe → fakesink
    nvurisrcbin ─┘

Mọi tham số đọc từ `configs/pipeline/streams.yaml`, không hardcode (CLAUDE.md §8).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from common.config import load_yaml, resolve  # noqa: E402
from common.logging import get_logger  # noqa: E402

log = get_logger(__name__)

ProbeFn = Callable[[Gst.Pad, Gst.PadProbeInfo, object], Gst.PadProbeReturn]


class PipelineError(RuntimeError):
    """Không dựng được pipeline — thiếu plugin, sai config, hoặc link hỏng."""


@dataclass(slots=True, frozen=True)
class SourceSpec:
    """Một camera. `cam_id` là định danh ổn định, KHÔNG phải index của streammux."""

    cam_id: str
    uri: str


@dataclass(slots=True)
class StreammuxConfig:
    width: int = 1920
    height: int = 1080
    batched_push_timeout_us: int = 40_000
    live_source: bool = False
    attach_sys_ts: bool = True
    """Gắn wall-clock của hệ thống vào frame meta. Cần cho `ts_ms` khi nguồn là file."""


@dataclass(slots=True)
class TrackerConfig:
    ll_lib_file: str = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
    ll_config_file: str = "configs/pipeline/config_tracker_NvDCF_perf.yml"
    width: int = 960
    height: int = 544
    display_tracking_id: bool = False


@dataclass(slots=True)
class PipelineConfig:
    sources: list[SourceSpec] = field(default_factory=list)
    streammux: StreammuxConfig = field(default_factory=StreammuxConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    pgie_config_file: str = "configs/pipeline/config_infer_yolo11.txt"

    @property
    def cam_ids(self) -> list[str]:
        """Ánh xạ index pad của streammux → cam_id. Thứ tự trong YAML là thứ tự pad."""
        return [s.cam_id for s in self.sources]


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Đọc `configs/pipeline/streams.yaml`."""
    raw = load_yaml(path)

    sources = [
        SourceSpec(cam_id=str(s["cam_id"]), uri=str(s["uri"])) for s in raw.get("sources", [])
    ]
    if not sources:
        raise PipelineError(f"{path}: không có nguồn nào trong mục 'sources'")

    seen = {s.cam_id for s in sources}
    if len(seen) != len(sources):
        raise PipelineError(f"{path}: cam_id bị trùng — khoá toàn cục sẽ sai")

    cfg = PipelineConfig(sources=sources)
    for key, value in (raw.get("streammux") or {}).items():
        if hasattr(cfg.streammux, key):
            setattr(cfg.streammux, key, value)
    for key, value in (raw.get("tracker") or {}).items():
        if hasattr(cfg.tracker, key):
            setattr(cfg.tracker, key, value)
    if pgie := (raw.get("pgie") or {}).get("config_file"):
        cfg.pgie_config_file = str(pgie)
    return cfg


def _make(factory: str, name: str) -> Gst.Element:
    element = Gst.ElementFactory.make(factory, name)
    if element is None:
        raise PipelineError(
            f"không tạo được element '{factory}'. Plugin chưa cài, hoặc đang chạy "
            f"ngoài container DeepStream. Kiểm tra: gst-inspect-1.0 {factory}"
        )
    return element


def _link(src: Gst.Element, dst: Gst.Element) -> None:
    if not src.link(dst):
        raise PipelineError(f"không link được {src.get_name()} → {dst.get_name()}")


def _attach_source(pipeline: Gst.Pipeline, streammux: Gst.Element, index: int, spec: SourceSpec):
    """Tạo nvurisrcbin cho một nguồn và nối vào sink pad thứ `index` của streammux.

    nvurisrcbin tạo pad động (sau khi dò được định dạng), nên phải link trong callback
    'pad-added' chứ không link được ngay lúc dựng.
    """
    src_bin = _make("nvurisrcbin", f"source-{index}")
    src_bin.set_property("uri", spec.uri)
    # Với RTSP: tự kết nối lại khi mất luồng thay vì làm sập cả pipeline.
    # property chỉ có ở nguồn RTSP / bản DeepStream mới hơn — bỏ qua nếu không có.
    for prop, value in (("rtsp-reconnect-interval", 30), ("latency", 200)):
        with contextlib.suppress(TypeError):
            src_bin.set_property(prop, value)

    sink_pad = streammux.request_pad_simple(f"sink_{index}")
    if sink_pad is None:
        raise PipelineError(f"streammux không cấp được sink_{index}")

    def on_pad_added(_bin: Gst.Element, pad: Gst.Pad) -> None:
        if sink_pad.is_linked():
            return
        if pad.link(sink_pad) != Gst.PadLinkReturn.OK:
            log.error("không nối được %s vào sink_%d", spec.cam_id, index)
        else:
            log.info("nguồn %s (%s) đã nối vào sink_%d", spec.cam_id, spec.uri, index)

    src_bin.connect("pad-added", on_pad_added)
    pipeline.add(src_bin)
    return src_bin


def build_pipeline(cfg: PipelineConfig, probe: ProbeFn | None = None) -> Gst.Pipeline:
    """Dựng pipeline hoàn chỉnh. `probe` gắn vào src pad của nvtracker."""
    Gst.init(None)

    pipeline = Gst.Pipeline.new("mct-pipeline")
    if pipeline is None:
        raise PipelineError("không tạo được Gst.Pipeline")

    streammux = _make("nvstreammux", "streammux")
    streammux.set_property("batch-size", len(cfg.sources))
    streammux.set_property("width", cfg.streammux.width)
    streammux.set_property("height", cfg.streammux.height)
    streammux.set_property("batched-push-timeout", cfg.streammux.batched_push_timeout_us)
    streammux.set_property("live-source", cfg.streammux.live_source)
    try:
        streammux.set_property("attach-sys-ts", cfg.streammux.attach_sys_ts)
    except TypeError:
        log.warning("streammux không có 'attach-sys-ts' — ts_ms sẽ lấy từ wall clock của probe")
    pipeline.add(streammux)

    for index, spec in enumerate(cfg.sources):
        _attach_source(pipeline, streammux, index, spec)

    pgie = _make("nvinfer", "primary-gie")
    pgie.set_property("config-file-path", str(resolve(cfg.pgie_config_file)))
    pgie.set_property("batch-size", len(cfg.sources))

    tracker = _make("nvtracker", "tracker")
    tracker.set_property("ll-lib-file", cfg.tracker.ll_lib_file)
    tracker.set_property("ll-config-file", str(resolve(cfg.tracker.ll_config_file)))
    tracker.set_property("tracker-width", cfg.tracker.width)
    tracker.set_property("tracker-height", cfg.tracker.height)
    with contextlib.suppress(TypeError):
        tracker.set_property("display-tracking-id", cfg.tracker.display_tracking_id)

    sink = _make("fakesink", "sink")
    sink.set_property("sync", False)
    sink.set_property("async", False)

    for element in (pgie, tracker, sink):
        pipeline.add(element)

    _link(streammux, pgie)
    _link(pgie, tracker)
    _link(tracker, sink)

    if probe is not None:
        src_pad = tracker.get_static_pad("src")
        if src_pad is None:
            raise PipelineError("nvtracker không có src pad")
        src_pad.add_probe(Gst.PadProbeType.BUFFER, probe, None)

    log.info(
        "pipeline: %d nguồn, streammux %dx%d, pgie=%s, tracker=%s",
        len(cfg.sources),
        cfg.streammux.width,
        cfg.streammux.height,
        Path(cfg.pgie_config_file).name,
        Path(cfg.tracker.ll_config_file).name,
    )
    return pipeline
