"""Ánh xạ điểm chân trong ảnh về mặt phẳng mặt đất chung (hiện thực `GroundMapper`).

Với cặp camera **chồng lấn**, ngoại hình không phải bằng chứng mạnh nhất — vị trí mới là.
Hai người mặc đồ giống nhau vẫn phân biệt được nếu một người đứng ở góc quảng trường còn
người kia ở giữa; ngược lại, cùng một người nhìn từ hai camera thì hai điểm chân của họ
phải rơi gần như trùng nhau sau khi chiếu về mặt phẳng chung. Đó chính là thành phần
`λ · d_ground` mà `affinity.py` đã chừa sẵn chỗ (CLAUDE.md §6 bước 3).

**Giả thiết duy nhất:** người đứng trên một mặt phẳng. Khi đó điểm chân (đáy-giữa bbox,
`Detection.ground_point`) và điểm tương ứng trên mặt đất liên hệ với nhau bằng một phép
biến đổi xạ ảnh 3x3 — không cần biết nội/ngoại tham số camera, chỉ cần ≥4 cặp điểm tương
ứng. Đây là lý do chọn homography thay vì hiệu chỉnh camera đầy đủ: rẻ hơn nhiều, sai số
đủ dùng, và giải thích được trong báo cáo bằng đúng một công thức.

Ba phần:

  - `estimate_homography()` — ước lượng H từ các cặp điểm bằng **DLT chuẩn hoá** (Hartley),
    kèm vài vòng cắt tỉa điểm sai lớn để chịu được nhiễu chú thích;
  - `CameraHomography` — H của một camera + siêu dữ liệu (sai số, độ phân giải lúc hiệu
    chỉnh, tên mặt phẳng), đọc/ghi YAML trong `configs/cameras/homography/`;
  - `HomographyMapper` — gom nhiều camera, hiện thực giao thức `GroundMapper`.

Chỉ dùng numpy: `src/mct/` không được phụ thuộc OpenCV/GPU (CLAUDE.md §2 quy tắc 1), và
`cv2.findHomography` không đem lại gì mà 40 dòng DLT không làm được ở quy mô này.

**Toạ độ ảnh phải theo độ phân giải GỐC của camera** — đúng quy ước bbox ở CLAUDE.md §5.
H hiệu chỉnh ở 1920x1080 mà đem áp cho điểm theo toạ độ streammux 1280x720 thì sai lệch
âm thầm; `image_size` được ghi vào file hiệu chỉnh chính là để bắt lỗi đó
(`HomographyMapper.check_frame_size`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common.logging import get_logger

log = get_logger(__name__)

Point = tuple[float, float]

MIN_CORRESPONDENCES = 4
"""Homography có 8 bậc tự do; mỗi cặp điểm cho 2 phương trình → tối thiểu 4 cặp."""

_DEGENERATE_W = 1e-9
"""|w| nhỏ hơn ngưỡng này = điểm nằm trên đường chân trời, chiếu ra vô cực."""


# --------------------------------------------------------------------------- ước lượng


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chuẩn hoá Hartley: dời trọng tâm về gốc, co giãn cho khoảng cách trung bình = sqrt(2).

    Bỏ qua bước này thì DLT rất kém ổn định về số học: toạ độ pixel cỡ 10^3 trộn với toạ
    độ mét cỡ 10^0 làm ma trận A có số điều kiện tệ, và nghiệm SVD lệch hẳn.
    """
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_dist = float(np.sqrt((shifted**2).sum(axis=1)).mean())
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return shifted * scale, transform


def _dlt(image_pts: np.ndarray, world_pts: np.ndarray) -> np.ndarray:
    """DLT thuần: dựng ma trận A (2n x 9), lấy vector kỳ dị nhỏ nhất."""
    n = len(image_pts)
    a = np.zeros((2 * n, 9), dtype=np.float64)
    for i, ((u, v), (x, y)) in enumerate(zip(image_pts, world_pts, strict=True)):
        a[2 * i] = (-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x)
        a[2 * i + 1] = (0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y)
    _, _, vt = np.linalg.svd(a)
    return vt[-1].reshape(3, 3)


def apply_homography(matrix: np.ndarray, point: Point) -> Point | None:
    """Chiếu một điểm ảnh về mặt phẳng tham chiếu. `None` = điểm suy biến (chân trời)."""
    vec = matrix @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    if abs(vec[2]) < _DEGENERATE_W:
        return None
    return (float(vec[0] / vec[2]), float(vec[1] / vec[2]))


def project_points(matrix: np.ndarray, image_pts: np.ndarray) -> np.ndarray:
    """Bản vector hoá của `apply_homography` (dùng khi tính sai số hiệu chỉnh)."""
    homogeneous = np.hstack([image_pts, np.ones((len(image_pts), 1))])
    projected = homogeneous @ matrix.T
    w = np.where(np.abs(projected[:, 2:3]) < _DEGENERATE_W, np.nan, projected[:, 2:3])
    return projected[:, :2] / w


def reprojection_errors(
    matrix: np.ndarray, image_pts: np.ndarray, world_pts: np.ndarray
) -> np.ndarray:
    """Sai số từng điểm (mét) — vô cùng lớn cho điểm chiếu ra vô cực."""
    projected = project_points(matrix, image_pts)
    err = np.sqrt(((projected - world_pts) ** 2).sum(axis=1))
    return np.where(np.isfinite(err), err, np.inf)


@dataclass(slots=True, frozen=True)
class HomographyFit:
    """Kết quả ước lượng: H + thống kê sai số để biết có tin được không."""

    matrix: np.ndarray
    n_points: int
    """Số cặp điểm còn lại sau khi cắt tỉa."""

    n_input: int
    rmse_m: float
    median_err_m: float
    p95_err_m: float

    def summary(self) -> str:
        return (
            f"{self.n_points}/{self.n_input} điểm, RMSE={self.rmse_m:.3f} m, "
            f"trung vị={self.median_err_m:.3f} m, p95={self.p95_err_m:.3f} m"
        )


def estimate_homography(
    image_pts: Sequence[Point] | np.ndarray,
    world_pts: Sequence[Point] | np.ndarray,
    *,
    trim_ratio: float = 0.1,
    trim_rounds: int = 2,
) -> HomographyFit:
    """Ước lượng H (ảnh → mặt phẳng mét) bằng DLT chuẩn hoá, có cắt tỉa điểm sai lớn.

    Vì sao cần cắt tỉa: khi hiệu chỉnh bằng chú thích tự động (điểm chân = đáy-giữa bbox),
    một phần bbox bị cắt ở mép khung hoặc bị che nên điểm chân không nằm trên mặt đất.
    Bình phương tối thiểu thuần thì vài điểm hỏng đó kéo lệch cả H. Cắt `trim_ratio` phần
    tệ nhất rồi ước lượng lại — rẻ hơn RANSAC và đủ dùng khi tỉ lệ ngoại lai thấp
    (dữ liệu ở đây là chú thích tay, không phải detector).
    """
    img = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    world = np.asarray(world_pts, dtype=np.float64).reshape(-1, 2)
    if len(img) != len(world):
        raise ValueError(f"số điểm ảnh ({len(img)}) khác số điểm mặt đất ({len(world)})")
    if len(img) < MIN_CORRESPONDENCES:
        raise ValueError(f"cần ít nhất {MIN_CORRESPONDENCES} cặp điểm, nhận {len(img)}")
    if not 0.0 <= trim_ratio < 0.5:
        raise ValueError(f"trim_ratio phải trong [0, 0.5), nhận {trim_ratio}")

    n_input = len(img)
    keep = np.arange(n_input)
    matrix = _fit_once(img, world)

    for _ in range(trim_rounds if trim_ratio > 0.0 else 0):
        errors = reprojection_errors(matrix, img[keep], world[keep])
        n_keep = max(MIN_CORRESPONDENCES, round(len(keep) * (1.0 - trim_ratio)))
        if n_keep >= len(keep):
            break
        keep = keep[np.argsort(errors)[:n_keep]]
        matrix = _fit_once(img[keep], world[keep])

    errors = reprojection_errors(matrix, img[keep], world[keep])
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        raise ValueError("ước lượng thất bại: mọi điểm đều chiếu ra vô cực (điểm suy biến?)")
    return HomographyFit(
        matrix=matrix,
        n_points=len(keep),
        n_input=n_input,
        rmse_m=float(np.sqrt((finite**2).mean())),
        median_err_m=float(np.median(finite)),
        p95_err_m=float(np.percentile(finite, 95)),
    )


def _fit_once(img: np.ndarray, world: np.ndarray) -> np.ndarray:
    img_n, t_img = _normalize_points(img)
    world_n, t_world = _normalize_points(world)
    matrix = np.linalg.inv(t_world) @ _dlt(img_n, world_n) @ t_img
    if abs(matrix[2, 2]) < _DEGENERATE_W:
        raise ValueError("ước lượng thất bại: H[2,2] ~ 0, các điểm có thể thẳng hàng")
    return matrix / matrix[2, 2]


# --------------------------------------------------------------------- một camera


@dataclass(slots=True)
class CameraHomography:
    """H của một camera + siêu dữ liệu cần để dùng lại đúng."""

    cam_id: str
    matrix: np.ndarray
    """3x3, ảnh (pixel, độ phân giải gốc) → mặt phẳng tham chiếu (mét)."""

    plane: str = "ground"
    """Tên mặt phẳng tham chiếu. Hai camera khác `plane` thì KHÔNG so được với nhau."""

    image_size: tuple[int, int] | None = None
    """(width, height) lúc hiệu chỉnh — để phát hiện bbox theo độ phân giải khác."""

    rmse_m: float | None = None
    n_points: int | None = None
    source: str = ""

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        if self.matrix.shape != (3, 3):
            raise ValueError(f"{self.cam_id}: ma trận phải là 3x3, nhận {self.matrix.shape}")
        if not np.isfinite(self.matrix).all():
            raise ValueError(f"{self.cam_id}: ma trận chứa NaN/inf")
        if abs(np.linalg.det(self.matrix)) < 1e-12:
            raise ValueError(f"{self.cam_id}: ma trận suy biến (det ~ 0)")

    def project(self, point: Point) -> Point | None:
        """Điểm chân trong ảnh → (X, Y) mét. `None` nếu suy biến."""
        return apply_homography(self.matrix, point)

    def ground_polygon(self, **kwargs: Any) -> list[Point]:
        """Vùng mặt đất camera này nhìn thấy và định vị được. Rỗng nếu chưa biết `image_size`."""
        if self.image_size is None:
            return []
        return ground_polygon(self.matrix, self.image_size, **kwargs)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, cam_id: str | None = None
    ) -> CameraHomography:
        cam = str(data.get("cam_id", cam_id or ""))
        if not cam:
            raise ValueError("thiếu cam_id trong file hiệu chỉnh")
        size = data.get("image_size")
        return cls(
            cam_id=cam,
            matrix=np.array(data["matrix"], dtype=np.float64),
            plane=str(data.get("plane", "ground")),
            image_size=(int(size[0]), int(size[1])) if size else None,
            rmse_m=float(data["rmse_m"]) if data.get("rmse_m") is not None else None,
            n_points=int(data["n_points"]) if data.get("n_points") is not None else None,
            source=str(data.get("source", "")),
        )

    def to_mapping(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "cam_id": self.cam_id,
            "plane": self.plane,
            "unit": "m",
            "matrix": [[float(v) for v in row] for row in self.matrix],
        }
        if self.image_size is not None:
            data["image_size"] = [int(self.image_size[0]), int(self.image_size[1])]
        if self.rmse_m is not None:
            data["rmse_m"] = round(float(self.rmse_m), 6)
        if self.n_points is not None:
            data["n_points"] = int(self.n_points)
        if self.source:
            data["source"] = self.source
        return data

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_mapping(), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> CameraHomography:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_mapping(data, cam_id=path.stem)


# ------------------------------------------------- vùng phủ trên mặt phẳng


def _clip_halfplane(
    polygon: list[Point], normal: tuple[float, float, float], eps: float = 0.0
) -> list[Point]:
    """Cắt đa giác lồi bằng nửa mặt phẳng `a·x + b·y + c >= eps` (Sutherland–Hodgman)."""
    a, b, c = normal
    out: list[Point] = []
    n = len(polygon)
    for i in range(n):
        cur, nxt = polygon[i], polygon[(i + 1) % n]
        d_cur = a * cur[0] + b * cur[1] + c - eps
        d_nxt = a * nxt[0] + b * nxt[1] + c - eps
        if d_cur >= 0:
            out.append(cur)
        if (d_cur >= 0) != (d_nxt >= 0):
            t = d_cur / (d_cur - d_nxt)
            out.append((cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1])))
    return out


def ground_scale_m_per_px(matrix: np.ndarray, point: Point, *, dv: float = 1.0) -> float:
    """Một pixel theo chiều dọc ở `point` phủ bao nhiêu mét trên mặt đất.

    Đây là thước đo "còn định vị được không": sát chân camera một pixel là vài milimét,
    càng lên phía đường chân trời thì một pixel càng phủ rộng, tới mức điểm chân của một
    người không còn xác định nổi vị trí. Trả `inf` khi điểm nằm sau mặt phẳng camera.
    """
    a = apply_homography(matrix, point)
    b = apply_homography(matrix, (point[0], point[1] - dv))
    if a is None or b is None:
        return float("inf")
    return float(np.hypot(a[0] - b[0], a[1] - b[1]) / dv)


def ground_polygon(
    matrix: np.ndarray,
    image_size: tuple[int, int],
    *,
    max_ground_per_pixel_m: float = 0.05,
    max_range_m: float = 200.0,
) -> list[Point]:
    """Vùng mặt đất mà một camera nhìn thấy **và định vị được**, trên mặt phẳng tham chiếu.

    Đây là "sơ đồ camera" của dashboard (CLAUDE.md §3), lấy được miễn phí từ chính
    homography — không cần đo đạc gì thêm.

    **Cắt ở đâu mới đúng?** Cắt theo đường chân trời là sai về mặt hữu dụng: ngay dưới
    đường chân trời một pixel đã phủ hàng chục mét, nên đa giác kéo dài gần như vô tận và
    bản đồ thu vùng thật sự có người xuống thành một chấm (đo trên WildTrack: khung nhìn
    ra 144x144 m cho một quảng trường 12x36 m). Cắt ở bán kính cố định thì lại là con số
    tuỳ tiện, mỗi hiện trường một khác.

    Tiêu chí dùng ở đây có ý nghĩa vật lý: **cắt tại hàng ảnh mà một pixel bắt đầu phủ
    quá `max_ground_per_pixel_m` mét mặt đất**. Xa hơn mức đó thì dù có nhìn thấy người,
    điểm chân của họ cũng không định vị nổi, nên vẽ ra chỉ gây hiểu nhầm.

    Mặc định 0.05 m/pixel — "lệch một pixel là lệch 5 cm", nhỏ hơn hẳn ngưỡng
    `max_ground_dist_m` (1.0 m) mà engine dùng để loại cặp. Con số này còn có một xác nhận
    thực nghiệm dễ chịu: trên WildTrack, vùng phủ tính ở 0.05 m/pixel ra X [-7.3, 12.8],
    Y [-8.2, 22.4] m, gần trùng với vùng mà người chú thích dataset đã chọn để gán nhãn
    (X [-3.0, 9.0], Y [-7.5, 23.1] m). Nới lên 0.25 thì khung nhìn phình gấp ba mà không
    thêm được vùng nào có người.

    Ước lượng theo **cột giữa ảnh**: tỉ lệ còn thay đổi theo phương ngang (ống kính rộng),
    nhưng đây là dữ liệu để VẼ chứ không phải để tính toán, và cột giữa là đại diện tốt.

    `max_range_m` chỉ là chốt chặn cuối cho trường hợp suy biến.
    """
    width, height = float(image_size[0]), float(image_size[1])
    u_center = width / 2.0

    # `w <= 0` ở hàng đáy = mặt phẳng nằm SAU camera. Phép chiếu vẫn ra số (chỉ là bị lộn
    # ngược), nên phải kiểm riêng — dựa vào `apply_homography` trả None thì không bắt được.
    if float(matrix[2] @ np.array([u_center, height, 1.0])) <= 0.0:
        return []
    if ground_scale_m_per_px(matrix, (u_center, height)) > max_ground_per_pixel_m:
        return []  # ngay sát chân camera đã không định vị nổi

    step = max(1.0, height / 200.0)
    v_cut = height
    v = height - step
    while v >= 0.0:
        if ground_scale_m_per_px(matrix, (u_center, v)) > max_ground_per_pixel_m:
            break
        v_cut = v
        v -= step

    rect: list[Point] = [(0.0, v_cut), (width, v_cut), (width, height), (0.0, height)]
    projected = [p for point in rect if (p := apply_homography(matrix, point)) is not None]
    if len(projected) < 3:
        return []

    limit = float(max_range_m)
    for normal in ((1.0, 0.0, limit), (-1.0, 0.0, limit), (0.0, 1.0, limit), (0.0, -1.0, limit)):
        projected = _clip_halfplane(projected, normal)
        if len(projected) < 3:
            return []
    return projected


# ------------------------------------------------------------------- nhiều camera


@dataclass(slots=True)
class HomographyMapper:
    """Hiện thực `affinity.GroundMapper` cho một tập camera đã hiệu chỉnh.

    Camera chưa hiệu chỉnh không phải lỗi: `distance_m` trả `None` và affinity bỏ qua
    thành phần hình học cho cặp đó. Nhờ vậy hệ thống chạy được khi mới hiệu chỉnh một
    phần — đúng tình huống triển khai thật, camera nào rảnh thì đo trước.
    """

    cameras: dict[str, CameraHomography] = field(default_factory=dict)

    def __post_init__(self) -> None:
        planes = {cam.plane for cam in self.cameras.values()}
        if len(planes) > 1:
            raise ValueError(
                "các camera hiệu chỉnh về những mặt phẳng khác nhau "
                f"({sorted(planes)}) — khoảng cách giữa chúng vô nghĩa"
            )

    # ------------------------------------------------------------------ nạp

    @classmethod
    def load(cls, path: str | Path) -> HomographyMapper:
        """Nạp từ một thư mục `*.yaml` (mỗi camera một file) hoặc một file gộp.

        File gộp có dạng `cameras: {cam01: {matrix: ...}, ...}` — tiện khi số camera ít.
        """
        path = Path(path)
        if path.is_dir():
            cams = [CameraHomography.load(p) for p in sorted(path.glob("*.yaml"))]
            if not cams:
                raise FileNotFoundError(f"{path} không có file hiệu chỉnh *.yaml nào")
            return cls({cam.cam_id: cam for cam in cams})

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = data.get("cameras")
        if entries is None:  # file của đúng một camera
            cam = CameraHomography.from_mapping(data, cam_id=path.stem)
            return cls({cam.cam_id: cam})
        return cls(
            {
                cam_id: CameraHomography.from_mapping(entry, cam_id=cam_id)
                for cam_id, entry in entries.items()
            }
        )

    @classmethod
    def from_fits(
        cls,
        fits: Mapping[str, HomographyFit],
        *,
        plane: str = "ground",
        image_size: tuple[int, int] | None = None,
        source: str = "",
    ) -> HomographyMapper:
        return cls(
            {
                cam_id: CameraHomography(
                    cam_id=cam_id,
                    matrix=fit.matrix,
                    plane=plane,
                    image_size=image_size,
                    rmse_m=fit.rmse_m,
                    n_points=fit.n_points,
                    source=source,
                )
                for cam_id, fit in fits.items()
            }
        )

    def save(self, directory: str | Path) -> list[Path]:
        return [
            cam.save(Path(directory) / f"{cam_id}.yaml")
            for cam_id, cam in sorted(self.cameras.items())
        ]

    # ------------------------------------------------------------------ dùng

    def has(self, cam_id: str) -> bool:
        return cam_id in self.cameras

    @property
    def calibrated(self) -> list[str]:
        return sorted(self.cameras)

    def project(self, cam_id: str, point: Point) -> Point | None:
        """Điểm chân trong ảnh của `cam_id` → (X, Y) mét, hoặc `None` nếu chưa hiệu chỉnh."""
        cam = self.cameras.get(cam_id)
        return None if cam is None else cam.project(point)

    def distance_m(self, cam_a: str, point_a: Point, cam_b: str, point_b: Point) -> float | None:
        """Giao thức `GroundMapper`: khoảng cách hai điểm chân trên mặt phẳng chung.

        `None` khi một trong hai camera chưa hiệu chỉnh, hoặc điểm chiếu ra vô cực — khi
        đó affinity bỏ qua thành phần hình học thay vì đoán bừa một khoảng cách.
        """
        pa = self.project(cam_a, point_a)
        pb = self.project(cam_b, point_b)
        if pa is None or pb is None:
            return None
        return float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))

    def footprints(self, **kwargs: Any) -> dict[str, list[Point]]:
        """Vùng phủ mặt đất của từng camera — dữ liệu vẽ sơ đồ của dashboard."""
        return {
            cam_id: polygon
            for cam_id, cam in sorted(self.cameras.items())
            if (polygon := cam.ground_polygon(**kwargs))
        }

    def check_frame_size(self, cam_id: str, width: int, height: int) -> str | None:
        """Cảnh báo nếu khung hình đang chạy khác độ phân giải lúc hiệu chỉnh.

        Đây là biến thể của cạm bẫy toạ độ streammux (CLAUDE.md §5): sai lệch không gây
        lỗi, chỉ làm khoảng cách mặt đất sai âm thầm. Trả chuỗi mô tả nếu lệch.
        """
        cam = self.cameras.get(cam_id)
        if cam is None or cam.image_size is None:
            return None
        if (width, height) == cam.image_size:
            return None
        return (
            f"{cam_id}: hiệu chỉnh ở {cam.image_size[0]}x{cam.image_size[1]} nhưng khung hình "
            f"đang là {width}x{height} — điểm chân phải quy về độ phân giải gốc trước khi chiếu"
        )

    def warn_frame_sizes(self, sizes: Iterable[tuple[str, int, int]]) -> list[str]:
        """Chạy `check_frame_size` cho nhiều camera, log mọi cảnh báo. Trả danh sách."""
        warnings = [
            msg
            for cam_id, w, h in sizes
            if (msg := self.check_frame_size(cam_id, w, h)) is not None
        ]
        for msg in warnings:
            log.warning("%s", msg)
        return warnings
