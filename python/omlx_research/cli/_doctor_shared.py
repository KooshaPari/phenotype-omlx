"""Pure helpers used by ``omlx_research.cli.doctor``.

Kept in its own module so ``doctor.py`` can stay focused on the public
API and the individual check functions. Everything here is stdlib-only
and side-effect-free so it can be tested directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

#: Status value for a check that passed cleanly.
PASS: str = "pass"
#: Status value for a check that ran but produced a non-fatal issue.
WARN: str = "warn"
#: Status value for a check that failed and must be fixed before use.
FAIL: str = "fail"

#: Minimum Python version supported by the omlx-research CLI.
MIN_PYTHON: tuple[int, int] = (3, 14)

#: Documented SOTA coverage count for model-kernels KernelOp tags.
#: Bumped from 22 -> 24 when SlidingWindowAttention + sliding_window_attention
#: tag landed (Qwen3-Next long-context support).
EXPECTED_KERNEL_OP_COUNT: int = 24


# ---------------------------------------------------------------------------
# Record type shared with ``doctor.py``
# ---------------------------------------------------------------------------

@dataclass
class Check:
    """One row of the doctor report."""

    id: str
    description: str
    status: str
    details: str = ""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def project_root() -> str:
    """Return the absolute path to the phenotype-omlx repo root."""
    # this file: <root>/python/omlx_research/cli/_doctor_shared.py
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def read_cargo_version(crate_relpath: str) -> str:
    """Best-effort: read a crate's Cargo.toml ``version`` field.

    Resolves workspace inheritance (``version.workspace = true``) by
    walking up to the nearest ancestor ``Cargo.toml`` that defines a
    ``[workspace.package]`` table and reading its ``version`` field.
    Returns ``"unknown"`` if anything is unreadable so callers can
    surface it without crashing.
    """
    crate_dir = os.path.join(project_root(), crate_relpath)
    crate_toml = os.path.join(crate_dir, "Cargo.toml")
    if not os.path.isfile(crate_toml):
        return "unknown"
    try:
        with open(crate_toml, "r", encoding="utf-8") as f:
            crate_text = f.read()
    except OSError:
        return "unknown"

    # Direct crate-level version
    for line in crate_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version") and "=" in stripped and "workspace" not in stripped:
            value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value

    # Workspace inheritance
    workspace_candidates = [
        os.path.join(crate_dir, "Cargo.toml"),
        os.path.abspath(os.path.join(crate_dir, "..", "Cargo.toml")),
        os.path.abspath(os.path.join(crate_dir, "..", "..", "Cargo.toml")),
    ]
    for ws_path in workspace_candidates:
        if not os.path.isfile(ws_path):
            continue
        try:
            with open(ws_path, "r", encoding="utf-8") as f:
                ws_text = f.read()
        except OSError:
            continue
        if "[workspace.package]" not in ws_text:
            continue
        in_block = False
        for line in ws_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and stripped != "[workspace.package]":
                in_block = False
                continue
            if stripped.startswith("[workspace.package]"):
                in_block = True
                continue
            if in_block and stripped.startswith("version"):
                value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    return "unknown"


def read_abi_version() -> Optional[str]:
    """Read ``ABI_VERSION_CURRENT`` from ``perf-core/native-abi/src/version.rs``.

    Handles the canonical Rust struct-field syntax::

        pub const ABI_VERSION_CURRENT: AbiVersion = AbiVersion {
            major: 1,
            minor: 0,
        };

    Returns ``None`` if the file is missing or the constant cannot be
    parsed — callers should fall back to "unknown" with a warn status.
    """
    path = os.path.join(
        project_root(), "perf-core", "native-abi", "src", "version.rs"
    )
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None

    def _extract(field_name: str) -> Optional[int]:
        """Return the integer literal after ``field_name:`` or ``field_name =``."""
        for line in text.splitlines():
            stripped = line.strip().rstrip(",")
            if not stripped.startswith(field_name):
                continue
            for sep in (":", "="):
                idx = stripped.find(sep)
                if idx < 0:
                    continue
                tail = stripped[idx + 1:].strip().rstrip(",").strip()
                try:
                    return int(tail)
                except ValueError:
                    # Try the next separator (this line may use ':' or '=').
                    continue
        return None

    major = _extract("major")
    if major is None:
        return None
    minor = _extract("minor")
    return f"{major}.{minor}" if minor is not None else str(major)


def collect_kernel_op_tags() -> list[str]:
    """Parse ``perf-core/model-kernels/src/lib.rs`` for ``tag()`` match arms.

    Returns the raw right-hand string values (e.g. ``["dense_attention",
    "gqa_attention", ...]``). Empty list on any error.
    """
    path = os.path.join(
        project_root(), "perf-core", "model-kernels", "src", "lib.rs"
    )
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return []
    tags: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if "=>" not in stripped or "KernelOp::" not in stripped:
            continue
        try:
            rhs = stripped.split("=>", 1)[1]
        except IndexError:
            continue
        first_quote = rhs.find('"')
        if first_quote < 0:
            continue
        tail = rhs[first_quote + 1:]
        end_quote = tail.find('"')
        if end_quote < 0:
            continue
        tag = tail[:end_quote].strip()
        if tag:
            tags.append(tag)
    return tags
