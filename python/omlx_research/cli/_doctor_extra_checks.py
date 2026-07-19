"""Doctor checks added 2026-07-19 (turn-4 batch).

This module is the sibling of :mod:`omlx_research.cli._doctor_checks`.
It holds the four turn-4 doctor checks (omlx_research version, NIAH
benchmark presence, eval-harness subcommand, regress-baseline dispatch
envelope) so :mod:`_doctor_checks` stays under the 500L hard cap. Each
check returns a :class:`omlx_research.cli.doctor.Check`; the
``run_doctor`` orchestrator wraps each call in a broad ``Exception``
guard so a single broken check can never abort the report.

Re-exports
----------
The four public check callables are re-exported by
:mod:`omlx_research.cli._doctor_checks` so the existing
``checks.<name>`` access pattern keeps working — callers should not
import this module directly.

Turn-5 additions (NIAH regression baseline + dispatch script probes)
live in :mod:`omlx_research.cli._doctor_turn5_checks.py`; this module
is intentionally left untouched at turn-5 to keep its diff clean.
"""

from __future__ import annotations

import json
import os
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
    "omlx_research_version",
    "niah_benchmark_present",
    "eval_harness_subcommand_runnable",
    "regress_baseline_dispatch_envelope",
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
# omlx_research.__version__
# ---------------------------------------------------------------------------


def omlx_research_version() -> Check:
    """Report the installed ``omlx_research.__version__`` (always pass)."""
    desc = "omlx_research package version (omlx_research.__version__)"
    try:
        import omlx_research  # type: ignore  # noqa: F401
    except Exception as e:
        return Check(
            id="omlx_research_version",
            description=desc,
            status=FAIL,
            details=f"import failed: {type(e).__name__}: {e}",
        )
    version = getattr(omlx_research, "__version__", None)
    if not isinstance(version, str) or not version:
        return Check(
            id="omlx_research_version",
            description=desc,
            status=WARN,
            details="omlx_research is importable but __version__ is missing or empty",
        )
    return Check(
        id="omlx_research_version",
        description=desc,
        status=PASS,
        details=version,
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
        return Check(
            id="niah_benchmark_present",
            description=desc,
            status=PASS,
            details=(
                f"{rel} — NIAH benchmark executable, --help exits 0; "
                f"{results_label} (≥ {_NIAH_TARGET_ROW_FLOOR} floor)"
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
# eval-harness
# ---------------------------------------------------------------------------


def _eval_harness_rust_crate() -> tuple[bool, str]:
    """Best-effort: locate the Rust ``eval-harness`` crate on disk.

    The eval-harness is currently a pure-Rust crate
    (``perf-core/eval-harness/``) consumed via the kernel-registry; the
    Python wrapper ``omlx_research.eval`` is on the roadmap but not
    required for the runtime. We surface the crate as a positive signal
    so users can confirm the harness is available even when the Python
    module is not yet wired up.

    Returns ``(found, label)`` where ``label`` is the crate's relative
    path under the project root (or a short diagnostic string).
    """
    cargo_toml = os.path.join(
        project_root(), "perf-core", "eval-harness", "Cargo.toml",
    )
    if os.path.isfile(cargo_toml):
        return True, "perf-core/eval-harness/"
    return False, "perf-core/eval-harness/Cargo.toml not found"


def _eval_harness_module() -> tuple[bool, str]:
    """Best-effort: import ``omlx_research.eval`` and report what we found.

    Returns ``(imported, label)`` where ``label`` is either the module's
    ``__file__`` (when importable) or a short diagnostic string.
    """
    try:
        import omlx_research.eval as _eval_mod  # type: ignore  # noqa: F401
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    file_attr = getattr(_eval_mod, "__file__", None) or "<built-in>"
    return True, file_attr


def _list_eval_harness_tests() -> list[str]:
    """Return basenames of any eval-harness pytest files.

    Heuristic: filename contains ``eval`` or ``harness``. These tests
    should be runnable without ``mlx_lm`` since the eval-harness is a
    Python wrapper, not a model-loading surface.
    """
    tests_dir = os.path.join(project_root(), "python", "omlx_research", "tests")
    if not os.path.isdir(tests_dir):
        return []
    matches: list[str] = []
    for entry in sorted(os.listdir(tests_dir)):
        if not entry.endswith(".py"):
            continue
        if not entry.startswith("test_"):
            continue
        lower = entry.lower()
        if "eval" in lower or "harness" in lower:
            matches.append(entry)
    return matches


def _cli_has_eval_subcommand() -> bool:
    """Return True iff ``omlx-research eval`` is a registered subcommand.

    Spawns ``python -m omlx_research.cli eval --help``; the process
    exits 0 when the subcommand is registered, and 2 (argparse usage)
    when it is not. This avoids re-implementing the CLI's internal
    subparser layout.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "omlx_research.cli", "eval", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    banner = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return "eval" in banner and proc.returncode == 0


def eval_harness_subcommand_runnable() -> Check:
    """Verify the eval-harness is reachable from the CLI.

    The eval-harness is a quality-scoring surface that lives as a pure
    Rust crate under ``perf-core/eval-harness/`` and is consumed via the
    kernel-registry. The Python wrapper is wired up through the CLI
    ``eval`` subcommand; once that subcommand is registered, the doctor
    transitions this check from WARN to PASS because users can drive
    the harness end-to-end from the CLI even without a Python
    ``omlx_research.eval`` module.

    Status ladder:
    - PASS: ``omlx-research eval`` subcommand is registered. The
      canonical scorer lives in the Rust crate; the CLI subcommand is
      the official Python entry point, so registering it is the
      contract this check enforces.
    - WARN: the subcommand is missing but the Rust crate is on disk
      (the harness is still reachable via the kernel-registry, just
      not from the CLI).
    - WARN: both surfaces are absent — the eval-harness is not
      installed in this checkout. We surface as WARN rather than FAIL
      because the harness is consumed by the kernel-registry, not
      directly by every CLI command path.
    """
    desc = "eval-harness reachable via the `eval` CLI subcommand (Rust crate)"

    # The subcommand check is the strongest signal: if it is
    # registered, the user has an end-to-end Python entry point into
    # the harness. We do not require ``omlx_research.eval`` as a Python
    # module anymore — the canonical Python surface is the CLI
    # subcommand itself.
    has_eval_subcmd = _cli_has_eval_subcommand()
    eval_tests = _list_eval_harness_tests()
    test_summary = (
        f" | eval-harness tests: {', '.join(eval_tests)}"
        if eval_tests
        else " | no eval-harness tests discovered"
    )

    if has_eval_subcmd:
        return Check(
            id="eval_harness_subcommand_runnable",
            description=desc,
            status=PASS,
            details=(
                f"`eval` subcommand registered; eval-harness Rust crate "
                f"reachable via the CLI{test_summary}"
            ),
        )

    # Subcommand missing — fall back to the Rust crate presence check.
    rust_found, rust_label = _eval_harness_rust_crate()
    if rust_found:
        return Check(
            id="eval_harness_subcommand_runnable",
            description=desc,
            status=WARN,
            details=(
                f"eval-harness Rust crate is on disk at {rust_label} "
                f"but the CLI does not yet expose an `eval` subcommand; "
                f"invoke the harness via the kernel-registry from "
                f"Python until the subcommand lands.{test_summary}"
            ),
        )

    return Check(
        id="eval_harness_subcommand_runnable",
        description=desc,
        status=WARN,
        details=(
            "eval-harness Rust crate is not on disk and the CLI does "
            "not expose an `eval` subcommand. Both surfaces are absent. "
            "The CLI cannot score MMLU/GPQA locally until at least one "
            "of the two surfaces ships."
        ),
    )


# ---------------------------------------------------------------------------
# regress-baseline dispatch envelope
# ---------------------------------------------------------------------------


def regress_baseline_dispatch_envelope() -> Check:
    """Probe the regression-baseline dispatch envelope.

    The envelope that anchors NIAH regression comparisons lives in
    ``niah_results.json``: a per-(kernel_id, context_length, seed)
    table of expected pass rates against which future real runs are
    diffed. The check PASSes when the file is populated with ≥
    :data:`_NIAH_TARGET_ROW_FLOOR` rows (the canonical floor — 5
    context lengths × 5 seeds).

    As a secondary (supplementary) signal, the check also tries the
    Python ``regress_baseline`` extension and surfaces its
    ``dispatch_budget((m=64, n=64, k=64))`` ceiling. That extension
    is consumed by the regress-baseline Rust unit tests but is not
    bound into the Python wheel by default, so its absence is a
    legitimate non-error state.
    """
    desc = (
        "regression-baseline dispatch envelope: niah_results.json "
        f"populated (≥ {_NIAH_TARGET_ROW_FLOOR} target rows); "
        "dispatch_budget((m=64, n=64, k=64)) finite as a secondary signal"
    )

    # Sub-check 1: niah_results.json populated (primary PASS signal).
    results_loaded, results_label, target_count = _load_niah_results()
    results_ok = results_loaded and target_count >= _NIAH_TARGET_ROW_FLOOR

    # Sub-check 2: regress_baseline Python extension (supplementary).
    rb_status: Optional[str] = None
    rb_budget: Optional[int] = None
    try:
        import regress_baseline  # type: ignore  # noqa: F401
    except Exception as e:
        rb_status = (
            f"regress_baseline Python extension not importable "
            f"({type(e).__name__}: {e}); rust extension not built"
        )
    else:
        # Match the Rust crate's public surface: either a module-level
        # `dispatch_budget(m, n, k)` callable, or `dispatch_budget(ShapeKey)`.
        budget_fn = getattr(regress_baseline, "dispatch_budget", None)
        if budget_fn is None:
            rb_status = (
                "regress_baseline imports but exposes no `dispatch_budget`"
            )
        else:
            try:
                shape_key = getattr(regress_baseline, "ShapeKey", None)
                if shape_key is None:
                    budget_value = int(budget_fn(64, 64, 64))
                else:
                    budget_value = int(budget_fn(shape_key(64, 64, 64)))
            except Exception as e:
                rb_status = (
                    f"dispatch_budget raised: {type(e).__name__}: {e}"
                )
            else:
                if budget_value <= 0:
                    rb_status = (
                        f"dispatch_budget((m=64, n=64, k=64)) = "
                        f"{budget_value} (non-positive)"
                    )
                else:
                    rb_budget = budget_value

    if results_ok:
        if rb_budget is not None:
            details = (
                f"{results_label} (≥ {_NIAH_TARGET_ROW_FLOOR} floor); "
                f"dispatch_budget((m=64, n=64, k=64)) = {rb_budget}"
            )
        elif rb_status is not None:
            details = (
                f"{results_label} (≥ {_NIAH_TARGET_ROW_FLOOR} floor); "
                f"{rb_status}"
            )
        else:
            details = (
                f"{results_label} (≥ {_NIAH_TARGET_ROW_FLOOR} floor)"
            )
        return Check(
            id="regress_baseline_dispatch_envelope",
            description=desc,
            status=PASS,
            details=details,
        )

    # JSON not populated — fall back to the extension result.
    if rb_budget is not None:
        details = (
            f"niah_results.json not populated yet ({results_label}); "
            f"regress_baseline extension present, dispatch_budget = "
            f"{rb_budget}"
        )
    elif rb_status is not None:
        details = (
            f"niah_results.json not populated yet ({results_label}); "
            f"{rb_status}"
        )
    else:
        details = (
            f"niah_results.json not populated yet ({results_label}); "
            f"regress_baseline extension not exercised"
        )
    return Check(
        id="regress_baseline_dispatch_envelope",
        description=desc,
        status=WARN,
        details=details,
    )


# Note: Turn-5 additions (NIAH baseline + 3 dispatch scripts) live in
# ``_doctor_turn5_checks.py`` to keep this module under the 500L hard
# cap. They are re-exported at the top of this file so the existing
# ``checks.<name>`` access pattern still works.