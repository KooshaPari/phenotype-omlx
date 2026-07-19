"""NIAH (needle-in-haystack) doctor checks added 2026-07-19.

This module is one of three siblings carved out of the original
:mod:`omlx_research.cli._doctor_extra_checks` module (turn-9 split,
module-size sweep). It owns the NIAH benchmark check plus the
shared ``niah_results.json`` loader that other modules reuse.

Layout reminder
---------------
- :mod:`omlx_research.cli._doctor_extra_niah` — NIAH benchmark check
  + ``niah_results.json`` helpers
- :mod:`omlx_research.cli._doctor_extra_eval` — eval-harness check
- :mod:`omlx_research.cli._doctor_extra_kernel` — package version +
  regress-baseline dispatch envelope

The check returns a :class:`omlx_research.cli.doctor.Check`; the
``run_doctor`` orchestrator wraps each call in a broad ``Exception``
guard so a single broken check can never abort the report.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Optional

from ._doctor_shared import (
    PASS,
    WARN,
    Check,
    project_root,
)


__all__ = [
    "niah_benchmark_present",
    "_load_niah_results",
    "_find_niah_benchmark",
]


# ---------------------------------------------------------------------------
# niah_results.json helpers (shared between niah_benchmark_present and
# regress_baseline_dispatch_envelope).
#
# Both doctor checks now key off the populated niah_results.json target
# table that anchors the regression envelope. The floor is 25 rows
# (5 context lengths × 5 seeds), matching the documented benchmark sweep.
# ---------------------------------------------------------------------------

_NIAH_RESULTS_REL_PATH = "niah_results.json"
_NIAH_TARGET_ROW_FLOOR = 25  # 5 context lengths × 5 seeds
# Informational ceiling for the full envelope: 10 ctx × 5 seeds × 5 kernels.
# Not a PASS threshold — the WARN->PASS gate stays at the 25-row floor so
# the turn-9 regression contract does not regress silently. The ceiling is
# surfaced only in the PASS details string so the doctor report indicates
# when the envelope is fully expanded.
_NIAH_ENVELOPE_TARGET = 250  # 10 context lengths × 5 seeds × 5 kernels


def _load_niah_results() -> tuple[bool, str, int]:
    """Read ``niah_results.json`` defensively.

    Returns ``(loaded, label, target_count)``. ``loaded`` is True only
    when the file exists, parses as JSON with a dict root, contains a
    ``targets`` list, and the list has at least
    :data:`_NIAH_TARGET_ROW_FLOOR` entries. ``label`` is a short
    human-readable status string used in check details.
    """
    path = os.path.join(project_root(), _NIAH_RESULTS_REL_PATH)
    if not os.path.isfile(path):
        return False, f"{_NIAH_RESULTS_REL_PATH} not on disk", 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as e:
        return False, f"{_NIAH_RESULTS_REL_PATH}: {type(e).__name__}: {e}", 0
    if not isinstance(payload, dict):
        return (
            False,
            f"{_NIAH_RESULTS_REL_PATH} root is {type(payload).__name__}, "
            "expected dict",
            0,
        )
    targets = payload.get("targets")
    if not isinstance(targets, list):
        return (
            False,
            f"{_NIAH_RESULTS_REL_PATH}['targets'] is "
            f"{type(targets).__name__}, expected list",
            0,
        )
    return True, f"{_NIAH_RESULTS_REL_PATH} ({len(targets)} target rows)", len(targets)


# ---------------------------------------------------------------------------
# NIAH benchmark
# ---------------------------------------------------------------------------


def _find_niah_benchmark() -> Optional[str]:
    """Locate ``scripts/niah_benchmark.py`` (or a small set of alternates).

    Returns the absolute path to the script if it exists, otherwise ``None``.
    """
    root = project_root()
    candidates = [
        os.path.join(root, "scripts", "niah_benchmark.py"),
        os.path.join(root, "scripts", "niah.py"),
        os.path.join(root, "scripts", "needle_in_haystack.py"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def niah_benchmark_present() -> Check:
    """Confirm ``scripts/niah_benchmark.py`` is on disk and runs.

    Three sub-signals collapse into the same row:

    1. the script's existence on disk (warn if missing);
    2. ``python3 scripts/niah_benchmark.py --help`` returning exit 0
       (warn if not executable); and
    3. ``niah_results.json`` containing ≥
       :data:`_NIAH_TARGET_ROW_FLOOR` populated target rows (warn
       if absent — the JSON is the long-lived reference snapshot
       that future runs are diffed against).

    PASS requires both the script is executable AND the results
    file is populated — that is the canonical "ready to run a
    regression comparison" state.
    """
    desc = (
        "NIAH (needle-in-haystack) benchmark: script present + "
        "--help exits 0 + niah_results.json populated (≥ "
        f"{_NIAH_TARGET_ROW_FLOOR} target rows)"
    )
    script_path = _find_niah_benchmark()
    if script_path is None:
        return Check(
            id="niah_benchmark_present",
            description=desc,
            status=WARN,
            details=(
                "NIAH benchmark absent — needle-in-haystack coverage "
                "not exercisable via scripts/"
            ),
        )

    # Sub-check: the script must run with --help without crashing.
    try:
        proc = subprocess.run(
            [sys.executable, script_path, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return Check(
            id="niah_benchmark_present",
            description=desc,
            status=WARN,
            details=(
                f"found {os.path.relpath(script_path, project_root())} but "
                f"`--help` could not be invoked: {type(e).__name__}: {e}"
            ),
        )

    rel = os.path.relpath(script_path, project_root())
    help_ok = proc.returncode == 0

    # Sub-check: niah_results.json must have populated target rows.
    results_loaded, results_label, target_count = _load_niah_results()
    results_ok = results_loaded and target_count >= _NIAH_TARGET_ROW_FLOOR

    if help_ok and results_ok:
        envelope_note = (
            f"; envelope complete ({target_count}/{_NIAH_ENVELOPE_TARGET})"
            if target_count >= _NIAH_ENVELOPE_TARGET
            else f"; envelope partial ({target_count}/{_NIAH_ENVELOPE_TARGET} target)"
        )
        return Check(
            id="niah_benchmark_present",
            description=desc,
            status=PASS,
            details=(
                f"{rel} — NIAH benchmark executable, --help exits 0; "
                f"{results_label} (≥ {_NIAH_TARGET_ROW_FLOOR} floor)"
                + envelope_note
            ),
        )

    if help_ok and not results_ok:
        return Check(
            id="niah_benchmark_present",
            description=desc,
            status=WARN,
            details=(
                f"{rel} — NIAH benchmark executable, --help exits 0; "
                f"but niah_results.json is not populated yet "
                f"({results_label})"
            ),
        )

    err_tail = (proc.stderr or proc.stdout).strip().splitlines()[-1] if (
        proc.stderr or proc.stdout
    ) else ""
    return Check(
        id="niah_benchmark_present",
        description=desc,
        status=WARN,
        details=(
            f"{rel} present but `python3 {rel} --help` exited {proc.returncode}: "
            f"{err_tail[:200]}"
        ),
    )
