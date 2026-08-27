"""Đọc cấu hình: YAML trong configs/ + biến môi trường từ .env.

Không hardcode đường dẫn tuyệt đối, IP camera hay ngưỡng thuật toán trong code
(CLAUDE.md §8) — tất cả đi qua đây.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Thư mục gốc repo, tìm bằng cách đi ngược lên cho tới khi thấy pyproject.toml."""
    for candidate in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


@lru_cache(maxsize=1)
def _ensure_env_loaded() -> None:
    """Nạp .env một lần. Biến đã có sẵn trong môi trường luôn thắng file .env."""
    load_dotenv(project_root() / ".env", override=False)


def env(name: str, default: str | None = None) -> str | None:
    _ensure_env_loaded()
    return os.environ.get(name, default)


def redis_url() -> str:
    return env("REDIS_URL", "redis://localhost:6379/0") or "redis://localhost:6379/0"


def log_level() -> str:
    return (env("LOG_LEVEL", "INFO") or "INFO").upper()


def resolve(path: str | Path) -> Path:
    """Đường dẫn tương đối được hiểu là tương đối so với gốc repo, không phải cwd."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Đọc YAML, có expand ${BIEN_MOI_TRUONG} trong các giá trị chuỗi."""
    _ensure_env_loaded()
    full = resolve(path)
    if not full.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {full}")
    data = yaml.safe_load(full.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{full}: cấu hình phải là mapping ở cấp cao nhất")
    return _expand(data)


def _expand(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v) for v in node]
    if isinstance(node, str):
        return os.path.expandvars(node)
    return node
