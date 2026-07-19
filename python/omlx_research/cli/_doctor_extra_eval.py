"""eval-harness doctor checks added 2026-07-19.

This module is one of three siblings carved out of the original
:mod:`omlx_research.cli._doctor_extra_checks` module (turn-9 split,
module-size sweep). It owns the eval-harness subcommand reachability
check plus the eval-harness helpers (Rust crate probe, Python module
probe, pytest file discovery, CLI subcommand probe).

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

import os
import subprocess
import sys

from ._doctor_shared import (
    PASS,
    WARN,
    Check,
    project_root,
)


__all__ = [
    "eval_harness_subcommand_runnable",
]


# ---------------------------------------------------------------------------
# eval-harness helpers
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


# ---------------------------------------------------------------------------
# eval-harness subcommand check
# ---------------------------------------------------------------------------


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
