"""``doctor`` — runtime health diagnostics for the omlx-research stack.

Runs a fixed list of lightweight, read-only checks against the local
environment and prints a human-readable summary (or a JSON envelope
when ``--json`` is passed). Each check returns a record with ``id``,
``description``, ``status`` (``pass``/``warn``/``fail``), and
``details`` (str).

Exit codes:
    0 — every check is ``pass``
    1 — at least one check is ``warn`` or ``fail``

Implementation note: checks use only the standard library plus the
project's own files, so the doctor can run before optional dependencies
(``mlx``, ``mlx_lm``, ``_perf``, ``pytest`` extras) are installed. A
failing import is reported as ``warn`` — never raised.

Layout:
    doctor.py             — public API + orchestration + rendering (this file)
    _doctor_shared.py     — pure parsing helpers (Cargo.toml, version.rs, lib.rs)
    _doctor_checks.py     — individual check functions
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

from . import _doctor_checks as checks
from ._doctor_shared import (
    EXPECTED_KERNEL_OP_COUNT,
    FAIL,
    MIN_PYTHON,
    PASS,
    WARN,
    Check,
    project_root,
)


__all__ = [
    "PASS",
    "WARN",
    "FAIL",
    "MIN_PYTHON",
    "EXPECTED_KERNEL_OP_COUNT",
    "Check",
    "DoctorReport",
    "CHECKS",
    "run_doctor",
    "cmd_doctor",
    "register",
]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class DoctorReport:
    """Aggregate report — what ``run_doctor`` builds and ``cmd_doctor`` renders."""

    checks: list[Check] = field(default_factory=list)

    def exit_code(self) -> int:
        """0 if every check is pass; 1 if any check is warn or fail."""
        return 0 if all(c.status == PASS for c in self.checks) else 1

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "ok": all(c.status == PASS for c in self.checks),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "checks": [asdict(c) for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Check registry + runner
# ---------------------------------------------------------------------------

CHECKS: list[Callable[[], Check]] = [
    checks.python_version,
    checks.mlx_core,
    checks.mlx_lm,
    checks.turboquant_rust_extension,
    checks.kernel_registry_version,
    checks.regress_baseline_version,
    checks.model_kernels_operator_coverage,
    checks.native_abi_v1,
    checks.airlock_v2,
    checks.tests_runnable,
]


def run_doctor(_args: Optional[argparse.Namespace] = None) -> DoctorReport:
    """Run every check in order, returning a populated DoctorReport.

    Each check is wrapped in a broad ``Exception`` guard so a single
    broken check cannot abort the whole report — failures degrade to
    ``fail`` with the exception class name in the details.
    """
    report = DoctorReport()
    for check_fn in CHECKS:
        try:
            check = check_fn()
        except Exception as e:  # defensive only
            check = Check(
                id=getattr(check_fn, "__name__", "unknown"),
                description="(description unavailable — check raised)",
                status=FAIL,
                details=f"{type(e).__name__}: {e}",
            )
        report.checks.append(check)
    return report


# ---------------------------------------------------------------------------
# Rendering + CLI wiring
# ---------------------------------------------------------------------------

_STATUS_GLYPH = {PASS: "OK  ", WARN: "WARN", FAIL: "FAIL"}


def _render_human(report: DoctorReport) -> str:
    lines: list[str] = []
    lines.append("== omlx-research doctor ==")
    lines.append(
        f"python: {platform.python_version()}  platform: {platform.platform()}"
    )
    lines.append("")
    for c in report.checks:
        glyph = _STATUS_GLYPH.get(c.status, c.status.upper())
        lines.append(f"[{glyph}] {c.id}")
        lines.append(f"        {c.description}")
        if c.details:
            for detail_line in c.details.splitlines():
                lines.append(f"        {detail_line}")
    summary = (
        f"{sum(1 for c in report.checks if c.status == PASS)} pass, "
        f"{sum(1 for c in report.checks if c.status == WARN)} warn, "
        f"{sum(1 for c in report.checks if c.status == FAIL)} fail"
    )
    lines.append("")
    lines.append(f"summary: {summary}")
    return "\n".join(lines) + "\n"


def cmd_doctor(args: argparse.Namespace) -> int:
    """CLI entry point: ``doctor [--json]``."""
    report = run_doctor(args)
    if getattr(args, "json", False):
        sys.stdout.write(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
        )
    else:
        sys.stdout.write(_render_human(report))
    return report.exit_code()


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach the ``doctor`` subparser to the parent subparsers object."""
    p = subparsers.add_parser(
        "doctor",
        help="diagnose the runtime environment (Python, MLX, kernels, ABI, tests)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit a JSON envelope to stdout instead of the human summary",
    )
    p.set_defaults(fn=cmd_doctor)
