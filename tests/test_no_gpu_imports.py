"""Canh quy tắc bất biến số 1 của dự án (CLAUDE.md §2).

Chỉ `src/ds_pipeline` được phép import pyds/gi/tensorrt/pycuda/cuda. Mọi package khác
phải chạy được trên máy không GPU — đó là điều kiện để phát triển engine liên kết trên
Mac bằng fixture ghi sẵn. Một dòng import lạc chỗ là hỏng cả quy trình đó, và triệu
chứng sẽ chỉ hiện ra rất muộn, nên chặn ngay bằng test.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = {"pyds", "gi", "tensorrt", "pycuda", "cuda"}

# Các cây thư mục PHẢI chạy được khi không có GPU.
CPU_ONLY_ROOTS = ("src/common", "src/mct", "src/dashboard", "src/tools", "eval", "tests")

CPU_ONLY_PACKAGES = ("common", "mct", "dashboard", "tools")


def _python_files() -> list[Path]:
    files: list[Path] = []
    for rel in CPU_ONLY_ROOTS:
        root = REPO_ROOT / rel
        if root.is_dir():
            files.extend(sorted(root.rglob("*.py")))
    return files


def _imported_roots(tree: ast.AST) -> set[str]:
    """Tên module cấp cao nhất của mọi import tuyệt đối trong file."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_co_file_python_de_quet() -> None:
    files = _python_files()
    assert files, "Không tìm thấy file .py nào — kiểm tra lại CPU_ONLY_ROOTS"


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_khong_import_thu_vien_gpu(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offending = _imported_roots(tree) & FORBIDDEN
    assert not offending, (
        f"{path.relative_to(REPO_ROOT)} import {sorted(offending)} — "
        "chỉ src/ds_pipeline được phép. Xem CLAUDE.md §2, quy tắc bất biến 1."
    )


@pytest.mark.parametrize("pkg_name", CPU_ONLY_PACKAGES)
def test_import_duoc_khi_khong_co_gpu(pkg_name: str) -> None:
    """Quét tĩnh không bắt được import gián tiếp — thử import thật cho chắc."""
    pkg = importlib.import_module(pkg_name)
    for mod in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg_name}."):
        importlib.import_module(mod.name)


def test_ds_pipeline_khong_bi_quet() -> None:
    """Đảm bảo ds_pipeline nằm ngoài phạm vi — nó ĐƯỢC phép import pyds."""
    scanned = {p.relative_to(REPO_ROOT).as_posix() for p in _python_files()}
    assert not any(p.startswith("src/ds_pipeline/") for p in scanned)
