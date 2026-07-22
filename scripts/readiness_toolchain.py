"""Host toolchain discovery helpers for readiness gates."""

from __future__ import annotations

import os
import platform
from pathlib import Path


def host_triple() -> str | None:
    architecture = {"arm64": "aarch64", "aarch64": "aarch64", "amd64": "x86_64", "x86_64": "x86_64"}.get(platform.machine().lower())
    operating_system = {"darwin": "apple-darwin", "linux": "unknown-linux-gnu", "windows": "pc-windows-msvc"}.get(platform.system().lower())
    return f"{architecture}-{operating_system}" if architecture and operating_system else None


def is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)
