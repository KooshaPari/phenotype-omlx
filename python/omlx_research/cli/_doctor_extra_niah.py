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
import shutil
import subprocess
import sys
from typing import Optional

from ._doctor_shared import (
    FAIL,
    PASS,
    WARN,
    Check,
    project_root,
)


__all__ = [
    "niah_benchmark_present",
    "niah_benchmark_non_legacy_path",
    "julia_required_on_eval_path",
    "niah_instrumented_schema_v2_present",
    "_load_niah_results",
    "_find_niah_benchmark",
]

_INSTRUMENTED_NIAH_REL = "research/fr5_niah_qwen35_0_8b_instrumented.json"

_LEGACY_NIAH_PATH_MARKERS = (
    'REPO / "phenotype-omlx/python"',
    "REPO / 'phenotype-omlx/python'",
    'Path("/Users/kooshapari/CodeProjects/Phenotype/repos")',
)


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
    ev = payload.get("evidence_label")
    if isinstance(ev, str) and ev.strip():
        ev_note = f", evidence_label={ev!r}"
    else:
        ev_note = ", evidence_label=<missing>"
    return (
        True,
        f"{_NIAH_RESULTS_REL_PATH} ({len(targets)} target rows{ev_note})",
        len(targets),
    )


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


# ---------------------------------------------------------------------------
# FR-5 / E2 — reject legacy absolute phenotype-omlx/python path
# ---------------------------------------------------------------------------


def niah_benchmark_non_legacy_path() -> Check:
    """FAIL if ``scripts/niah_benchmark.py`` still embeds the legacy path.

    The absorbed-crate layout must import via repo-relative ``<root>/python``,
    never a hard-coded absolute ``…/repos`` + ``phenotype-omlx/python`` join.
    """
    desc = (
        "NIAH benchmark uses repo-relative python/ "
        "(no legacy absolute repos path join)"
    )
    script_path = _find_niah_benchmark()
    if script_path is None:
        return Check(
            id="niah_benchmark_non_legacy_path",
            description=desc,
            status=FAIL,
            details="NIAH benchmark script missing — cannot verify path layout",
        )
    try:
        with open(script_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        return Check(
            id="niah_benchmark_non_legacy_path",
            description=desc,
            status=FAIL,
            details=f"could not read {script_path}: {type(e).__name__}: {e}",
        )
    rel = os.path.relpath(script_path, project_root())
    hits = [m for m in _LEGACY_NIAH_PATH_MARKERS if m in text]
    if hits:
        return Check(
            id="niah_benchmark_non_legacy_path",
            description=desc,
            status=FAIL,
            details=(
                f"{rel} still embeds legacy marker {hits[0]!r} — use "
                'Path(__file__).resolve().parents[1] / "python"'
            ),
        )
    return Check(
        id="niah_benchmark_non_legacy_path",
        description=desc,
        status=PASS,
        details=f"{rel} — repo-relative python/ path (no legacy absolute join)",
    )


# ---------------------------------------------------------------------------
# FR-5 / E1 — Julia required on eval path (fail loud)
# ---------------------------------------------------------------------------


def julia_required_on_eval_path() -> Check:
    """FAIL if ``julia`` is not on PATH (FR-5 / toolchain policy).

    Presence is mandatory for the NIAH/eval path. Missing Julia is not a
    soft WARN — ship gates require fail-loud dependency signaling.
    """
    desc = "Julia required on NIAH/eval path (FR-5 E1; fail loud if missing)"
    julia_bin = shutil.which("julia")
    if julia_bin is None:
        return Check(
            id="julia_required_on_eval_path",
            description=desc,
            status=FAIL,
            details=(
                "julia not found on PATH — install Julia and ensure `julia` "
                "resolves before running NIAH/eval (no optional/stub path)"
            ),
        )
    version = ""
    try:
        proc = subprocess.run(
            [julia_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            version = (proc.stdout or proc.stderr or "").strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired) as e:
        return Check(
            id="julia_required_on_eval_path",
            description=desc,
            status=FAIL,
            details=(
                f"julia found at {julia_bin} but `--version` failed: "
                f"{type(e).__name__}: {e}"
            ),
        )
    details = f"{julia_bin}"
    if version:
        details = f"{julia_bin} ({version})"
    return Check(
        id="julia_required_on_eval_path",
        description=desc,
        status=PASS,
        details=details,
    )


# ---------------------------------------------------------------------------
# FR-5 / E3 — schema v2 instrumented compression envelope on disk
# ---------------------------------------------------------------------------


def niah_instrumented_schema_v2_present() -> Check:
    """FAIL if the schema-v2 instrumented NIAH envelope is missing or weak.

    Compression proof for FR-5 ship requires
    ``research/fr5_niah_qwen35_0_8b_instrumented.json`` with
    ``schema_version == 2`` plus ``packed_state_any`` and
    ``byte_reduction_any`` both true. Answer128 packs are exact-retrieval
    only and do not satisfy this check.
    """
    desc = (
        "Instrumented NIAH envelope is schema v2 with packed_state + "
        "byte_reduction (FR-5 compression proof)"
    )
    path = os.path.join(project_root(), _INSTRUMENTED_NIAH_REL)
    if not os.path.isfile(path):
        return Check(
            id="niah_instrumented_schema_v2_present",
            description=desc,
            status=FAIL,
            details=f"missing {_INSTRUMENTED_NIAH_REL}",
        )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        return Check(
            id="niah_instrumented_schema_v2_present",
            description=desc,
            status=FAIL,
            details=(
                f"could not load {_INSTRUMENTED_NIAH_REL}: "
                f"{type(e).__name__}: {e}"
            ),
        )
    if not isinstance(data, dict):
        return Check(
            id="niah_instrumented_schema_v2_present",
            description=desc,
            status=FAIL,
            details=f"{_INSTRUMENTED_NIAH_REL} root is not an object",
        )
    problems: list[str] = []
    if data.get("schema_version") != 2:
        problems.append(f"schema_version={data.get('schema_version')!r} (want 2)")
    if data.get("packed_state_any") is not True:
        problems.append(f"packed_state_any={data.get('packed_state_any')!r}")
    if data.get("byte_reduction_any") is not True:
        problems.append(f"byte_reduction_any={data.get('byte_reduction_any')!r}")
    if problems:
        return Check(
            id="niah_instrumented_schema_v2_present",
            description=desc,
            status=FAIL,
            details="; ".join(problems),
        )
    return Check(
        id="niah_instrumented_schema_v2_present",
        description=desc,
        status=PASS,
        details=(
            f"{_INSTRUMENTED_NIAH_REL} schema_version=2 "
            "packed_state_any=true byte_reduction_any=true"
        ),
    )
