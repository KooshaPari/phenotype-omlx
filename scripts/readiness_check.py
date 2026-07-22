#!/usr/bin/env python3
"""Deterministic phenotype-omlx readiness gate orchestration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Mapping, Sequence

try:
    import readiness_commands
    from readiness_commands import (
        CommandRunner,
        GateResult,
        PlannedCommand,
        ReadinessReport,
        _run_command_gate,
        _subprocess_runner,
        execute_planned_commands,
        resolve_cargo,
        wheel_abi_gate_plan,
    )
    from readiness_wheel import (
        _wheel_artifact,
        _wheel_gate_failure,
        _wheel_workspace,
        run_wheel_abi_contract_gate,
    )
except ModuleNotFoundError:
    from scripts import readiness_commands
    from scripts.readiness_commands import (
        CommandRunner,
        GateResult,
        PlannedCommand,
        ReadinessReport,
        _run_command_gate,
        _subprocess_runner,
        execute_planned_commands,
        resolve_cargo,
        wheel_abi_gate_plan,
    )
    from scripts.readiness_wheel import (
        _wheel_artifact,
        _wheel_gate_failure,
        _wheel_workspace,
        run_wheel_abi_contract_gate,
    )


WHEEL_IDENTITY_SCRIPT = readiness_commands.WHEEL_IDENTITY_SCRIPT


def run_source_python_gate(
    root: Path,
    runner: CommandRunner = _subprocess_runner,
    python_executable: str = sys.executable,
) -> GateResult:
    """Run only tests whose authority is the Python source tree."""

    tests = root / "python" / "tests"
    command = (
        python_executable,
        "-m",
        "pytest",
        str(tests / "test_mlx_backend.py"),
        str(tests / "test_readiness.py"),
        "-q",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "python")
    return _run_command_gate("source-python-contracts", command, root, runner, environment)


def run_readiness(
    root: Path,
    runner: CommandRunner = _subprocess_runner,
    cargo: str | None = None,
    include_python: bool = True,
    include_wheel_abi: bool = False,
) -> ReadinessReport:
    """Run readiness gates; full mode adds isolated release-wheel artifact proof."""

    resolved_root = root.expanduser().resolve()
    perf_core = resolved_root / "perf-core"
    cargo_bin = cargo or resolve_cargo()
    command_name = cargo_bin or "cargo"
    commands = (("cargo-check", (command_name, "check", "--workspace")), ("cargo-test", (command_name, "test", "--workspace")))
    if not perf_core.is_dir():
        return ReadinessReport(resolved_root, [GateResult(gate, "fail", "WORKSPACE_MISSING", str(perf_core), command) for gate, command in commands])
    if cargo_bin is None:
        return ReadinessReport(resolved_root, [GateResult(gate, "fail", "COMMAND_NOT_FOUND", "cargo", command) for gate, command in commands])
    results = [_run_command_gate(gate, command, perf_core, runner) for gate, command in commands]
    if include_python:
        results.append(run_source_python_gate(resolved_root, runner, sys.executable))
    if include_wheel_abi:
        results.append(run_wheel_abi_contract_gate(resolved_root, runner=runner, python_executable=sys.executable, maturin_executable=shutil.which("maturin") or "maturin"))
    else:
        results.append(GateResult("wheel-abi-contract", "not-run", "FULL_MODE_REQUIRED", "not verified in fast mode; rerun with --full for installed-wheel ABI proof", ["phenotype-omlx-ready", "--full"]))
    return ReadinessReport(resolved_root, results)


def _default_root() -> Path:
    configured = os.environ.get("PHENOTYPE_OMLX_HOME")
    return Path(configured) if configured else Path(__file__).resolve().parents[1]


def _print_human(report: ReadinessReport) -> None:
    for result in report.results:
        print(f"[ready] {result.gate:16s} {result.status:4s} {result.reason}: {result.detail}")
    summary = "OK" if report.fully_verified else "PARTIAL" if report.ok else "FAIL"
    print(f"[ready] {summary}: {len(report.results)} native gates")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_default_root())
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--full", action="store_true", help="build an isolated release wheel and verify its ABI/import contract")
    args = parser.parse_args(argv)
    report = run_readiness(args.root, include_wheel_abi=args.full)
    if args.json_output:
        print(json.dumps(report.as_dict(), sort_keys=True))
    else:
        _print_human(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
