"""Doctor checks added 2026-07-19 — turn 5 batch.

This module is the second sibling of :mod:`omlx_research.cli._doctor_checks`
and the turn-4 extra-checks split (now
:mod:`omlx_research.cli._doctor_extra_niah`,
:mod:`omlx_research.cli._doctor_extra_eval`,
:mod:`omlx_research.cli._doctor_extra_kernel`). It holds the four
turn-5 doctor checks (one NIAH baseline probe + three dispatch-script
probes) so the parent modules stay under the 500L hard cap.

Each check returns a :class:`omlx_research.cli.doctor.Check`; the
``run_doctor`` orchestrator wraps each call in a broad ``Exception``
guard so a single broken check can never abort the report.

Re-exports
----------
The four public check callables are re-exported by
:mod:`omlx_research.cli._doctor_checks` so the existing
``checks.<name>`` access pattern keeps working — callers should not
import this module directly.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

from ._doctor_registry import register_check
from ._doctor_shared import (
    FAIL,
    PASS,
    WARN,
    Check,
    project_root,
)


__all__ = [
    "niah_regression_baseline_exists",
    "dispatch_script_metal_exists",
    "dispatch_script_sglang_exists",
    "dispatch_script_vllm_exists",
]


# ---------------------------------------------------------------------------
# NIAH regression baseline
# ---------------------------------------------------------------------------


_NIAH_BASELINE_REL_PATH = "research/baselines/niah_baseline.json"


def _load_niah_baseline() -> tuple[bool, str]:
    """Load ``research/baselines/niah_baseline.json`` defensively.

    Returns ``(loaded, label)`` where ``label`` is a short diagnostic
    string (path on success, exception class + message on failure).
    """
    path = os.path.join(project_root(), _NIAH_BASELINE_REL_PATH)
    if not os.path.isfile(path):
        return False, f"{_NIAH_BASELINE_REL_PATH} not on disk"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, f"{_NIAH_BASELINE_REL_PATH} ({len(json.dumps(payload))} chars JSON)"


@register_check
def niah_regression_baseline_exists() -> Check:
    """Probe the seeded NIAH regression baseline.

    The baseline was seeded in turn-5 commit ``d9351dd`` and is the
    reference snapshot that future NIAH runs are compared against.
    Three sub-signals:

    - PASS: file is on disk AND parses as JSON AND has
      ``schema_version == 1`` AND ``kind == "niah_regression_baseline"``
    - WARN: file is missing (most common cause: a fresh checkout
      before the seed landed)
    - FAIL: file is on disk but does not parse, or has the wrong
      shape — that is a real bug in the seed

    The ``schema_version`` and ``kind`` checks are deliberately
    strict: a seed with the wrong schema would silently break every
    future NIAH regression run, so we escalate.
    """
    desc = (
        "NIAH regression baseline "
        "(research/baselines/niah_baseline.json) present + valid"
    )
    loaded, label = _load_niah_baseline()
    if not loaded:
        return Check(
            id="niah_regression_baseline_exists",
            description=desc,
            status=WARN,
            details=label,
        )
    # Re-load to inspect schema fields. Re-using the result would couple
    # too much; the cost of a second read is negligible.
    try:
        with open(
            os.path.join(project_root(), _NIAH_BASELINE_REL_PATH),
            "r",
            encoding="utf-8",
        ) as fh:
            payload = json.load(fh)
    except Exception as e:
        return Check(
            id="niah_regression_baseline_exists",
            description=desc,
            status=FAIL,
            details=(
                f"baseline read failed after first-pass OK: {type(e).__name__}: {e}"
            ),
        )
    if not isinstance(payload, dict):
        return Check(
            id="niah_regression_baseline_exists",
            description=desc,
            status=FAIL,
            details=f"baseline root is {type(payload).__name__}, expected dict",
        )
    schema_version = payload.get("schema_version")
    kind = payload.get("kind")
    if schema_version != 1:
        return Check(
            id="niah_regression_baseline_exists",
            description=desc,
            status=FAIL,
            details=f"baseline schema_version={schema_version!r}, expected 1",
        )
    if kind != "niah_regression_baseline":
        return Check(
            id="niah_regression_baseline_exists",
            description=desc,
            status=FAIL,
            details=f"baseline kind={kind!r}, expected 'niah_regression_baseline'",
        )
    return Check(
        id="niah_regression_baseline_exists",
        description=desc,
        status=PASS,
        details=label,
    )


# ---------------------------------------------------------------------------
# dispatch scripts
# ---------------------------------------------------------------------------


_DISPATCH_SCRIPT_BACKENDS: tuple[tuple[str, str], ...] = (
    ("metal", "scripts/dispatch/metal.sh"),
    ("sglang", "scripts/dispatch/sglang.sh"),
    ("vllm", "scripts/dispatch/vllm.sh"),
)


def _check_dispatch_script(backend: str, rel_path: str) -> Check:
    """Verify a single dispatch script is on disk + executable + --help exits 0.

    Returns PASS when the file exists, has at least one execute bit, and
    ``<rel_path> --help`` exits 0 within 5 seconds. WARN when any of
    the three conditions fail (file missing, no execute bit, --help
    non-zero exit).
    """
    check_id = f"dispatch_script_{backend}_exists"
    desc = f"dispatch script for backend '{backend}' ({rel_path}) present + executable"
    script_path = os.path.join(project_root(), rel_path)
    if not os.path.isfile(script_path):
        return Check(
            id=check_id,
            description=desc,
            status=WARN,
            details=f"{rel_path} not on disk — create it so the doctor can probe it",
        )
    if not os.access(script_path, os.X_OK):
        return Check(
            id=check_id,
            description=desc,
            status=WARN,
            details=f"{rel_path} is not executable (chmod +x missing)",
        )
    try:
        proc = subprocess.run(
            [script_path, "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return Check(
            id=check_id,
            description=desc,
            status=WARN,
            details=f"{rel_path} --help failed: {type(e).__name__}: {e}",
        )
    if proc.returncode != 0:
        return Check(
            id=check_id,
            description=desc,
            status=WARN,
            details=(
                f"{rel_path} --help exited {proc.returncode}: "
                f"{(proc.stderr or proc.stdout).strip()[:160]}"
            ),
        )
    return Check(
        id=check_id,
        description=desc,
        status=PASS,
        details=f"{rel_path} --help exits 0 (stub dispatch contract verified)",
    )


@register_check
def dispatch_script_metal_exists() -> Check:
    """Dispatch script for the Metal (Apple Silicon) backend."""
    return _check_dispatch_script("metal", "scripts/dispatch/metal.sh")


@register_check
def dispatch_script_sglang_exists() -> Check:
    """Dispatch script for the SGLang backend."""
    return _check_dispatch_script("sglang", "scripts/dispatch/sglang.sh")


@register_check
def dispatch_script_vllm_exists() -> Check:
    """Dispatch script for the vLLM backend."""
    return _check_dispatch_script("vllm", "scripts/dispatch/vllm.sh")
