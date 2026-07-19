"""Doctor checks added 2026-07-19.

This module is the sibling of :mod:`omlx_research.cli._doctor_checks`.
It holds the four newest doctor checks (omlx_research version, NIAH
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
"""

from __future__ import annotations

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

    Two sub-signals collapsed into the same row: the script's existence
    on disk (warn if missing) and ``python3 scripts/niah_benchmark.py
    --help`` returning exit 0 (warn if not executable). The check stays
    at WARN in both branches because NIAH is a benchmark, not a
    critical-path runtime dependency.
    """
    desc = "NIAH (needle-in-haystack) benchmark script present + executable"
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
    if proc.returncode == 0:
        return Check(
            id="niah_benchmark_present",
            description=desc,
            status=PASS,
            details=f"{rel} — NIAH benchmark executable, --help exits 0",
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
    """Verify the eval-harness Python wrapper imports cleanly.

    The eval-harness is a critical surface (used by ``omlx-research eval``
    once it ships), so an import failure escalates to ``FAIL``. A
    successful import without an ``eval`` subcommand is just a ``WARN``
    because the harness can still be imported directly.
    """
    desc = "omlx_research.eval (eval-harness) importable"
    imported, label = _eval_harness_module()
    if not imported:
        return Check(
            id="eval_harness_subcommand_runnable",
            description=desc,
            status=FAIL,
            details=(
                f"omlx_research.eval failed to import ({label}); the eval-harness "
                f"is a critical surface and must import cleanly. Check for missing "
                f"deps / syntax errors in omlx_research/eval/."
            ),
        )

    has_eval_subcmd = _cli_has_eval_subcommand()
    eval_tests = _list_eval_harness_tests()
    test_summary = (
        f" | eval-harness tests: {', '.join(eval_tests)}"
        if eval_tests
        else " | no eval-harness tests discovered"
    )

    if not has_eval_subcmd:
        return Check(
            id="eval_harness_subcommand_runnable",
            description=desc,
            status=WARN,
            details=(
                f"omlx_research.eval imports cleanly from {label} but the CLI "
                f"does not yet expose an `eval` subcommand; invoke the harness "
                f"via Python directly until the subcommand lands."
                f"{test_summary}"
            ),
        )

    return Check(
        id="eval_harness_subcommand_runnable",
        description=desc,
        status=PASS,
        details=(
            f"omlx_research.eval imports from {label}; `eval` subcommand "
            f"available{test_summary}"
        ),
    )


# ---------------------------------------------------------------------------
# regress-baseline dispatch envelope
# ---------------------------------------------------------------------------


def regress_baseline_dispatch_envelope() -> Check:
    """Probe ``regress_baseline.dispatch_budget`` for a small shape.

    Tries the Python extension if it is importable; otherwise returns
    WARN (the Rust extension may simply not have been built yet, and
    that is a legitimate state). When the extension is importable, we
    check the ceiling for ``(m=64, n=64, k=64)`` and require it to be
    a finite positive integer — exactly the same invariant the
    regress_baseline Rust unit tests enforce.
    """
    desc = (
        "regress_baseline Rust extension: dispatch_budget() finite "
        "for (m=64, n=64, k=64)"
    )
    try:
        import regress_baseline  # type: ignore  # noqa: F401
    except Exception as e:
        return Check(
            id="regress_baseline_dispatch_envelope",
            description=desc,
            status=WARN,
            details=(
                f"regress_baseline Python extension not importable "
                f"({type(e).__name__}: {e}); rust extension not built. "
                f"Run `maturin develop` from perf-core/regress-baseline "
                f"once a Python binding is added."
            ),
        )

    # Match the Rust crate's public surface: either a module-level
    # `dispatch_budget(m, n, k)` callable, or `dispatch_budget(ShapeKey)`.
    budget_fn = getattr(regress_baseline, "dispatch_budget", None)
    if budget_fn is None:
        return Check(
            id="regress_baseline_dispatch_envelope",
            description=desc,
            status=WARN,
            details=(
                "regress_baseline imports but exposes no `dispatch_budget`; "
                "Python bindings may be incomplete."
            ),
        )

    try:
        shape_key = getattr(regress_baseline, "ShapeKey", None)
        if shape_key is None:
            budget_value = int(budget_fn(64, 64, 64))
        else:
            budget_value = int(budget_fn(shape_key(64, 64, 64)))
    except Exception as e:
        return Check(
            id="regress_baseline_dispatch_envelope",
            description=desc,
            status=WARN,
            details=(
                f"dispatch_budget((m=64, n=64, k=64)) raised: "
                f"{type(e).__name__}: {e}"
            ),
        )

    if budget_value <= 0:
        return Check(
            id="regress_baseline_dispatch_envelope",
            description=desc,
            status=WARN,
            details=(
                f"dispatch_budget((m=64, n=64, k=64)) = {budget_value} — "
                f"expected a finite positive ceiling"
            ),
        )

    return Check(
        id="regress_baseline_dispatch_envelope",
        description=desc,
        status=PASS,
        details=f"dispatch_budget((m=64, n=64, k=64)) = {budget_value}",
    )