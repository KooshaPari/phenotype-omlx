"""Source-isolated wheel ABI readiness gate."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable, Mapping, Sequence

try:
    from readiness_commands import CommandRunner, GateResult, PlannedCommand, execute_planned_commands, resolve_cargo, wheel_abi_gate_plan, _subprocess_runner
except ModuleNotFoundError:
    from scripts.readiness_commands import CommandRunner, GateResult, PlannedCommand, execute_planned_commands, resolve_cargo, wheel_abi_gate_plan, _subprocess_runner


@contextmanager
def _wheel_workspace() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="phenotype-omlx-wheel-") as directory:
        yield Path(directory)


def _wheel_artifact(wheel_dirs: Sequence[Path]) -> Path | None:
    wheels = sorted(wheel for directory in wheel_dirs for wheel in directory.glob("*.whl"))
    return wheels[0] if len(wheels) == 1 else None


def _wheel_gate_failure(reason: str, detail: str, command: Sequence[str], evidence: Mapping[str, str] | None = None) -> GateResult:
    return GateResult("wheel-abi-contract", "fail", reason, detail, command, evidence=evidence)


def run_wheel_abi_contract_gate(root: Path, *, runner: CommandRunner = _subprocess_runner, python_executable: str = sys.executable, maturin_executable: str = "maturin", workspace_factory: Callable[[], object] = _wheel_workspace) -> GateResult:
    """Build and verify a release wheel from a disposable source-isolated environment."""
    resolved_root = root.expanduser().resolve()
    plan = wheel_abi_gate_plan(resolved_root, python_executable, maturin_executable)
    wheel_dirs = (resolved_root / "python" / "ffi" / "target" / "wheels",)
    cargo = resolve_cargo()
    if cargo is None:
        return _wheel_gate_failure("COMMAND_NOT_FOUND", "cargo", plan[0].command)
    build_environment = {"HOME": str(Path.home()), "PATH": os.pathsep.join((str(Path(cargo).parent), os.defpath))}
    build_plan = (PlannedCommand(plan[0].command, plan[0].cwd, build_environment),)
    with workspace_factory() as workspace:
        workspace_path = Path(workspace)
        venv = workspace_path / "venv"
        build = execute_planned_commands(build_plan, {}, runner)[0]
        if build.status != "pass":
            return _wheel_gate_failure(build.reason, build.detail, build.command)
        wheel = _wheel_artifact(wheel_dirs)
        if wheel is None:
            return _wheel_gate_failure("WHEEL_ARTIFACT_INVALID", "expected exactly one wheel in " + ", ".join(map(str, wheel_dirs)), build.command)
        results = execute_planned_commands(plan[1:], {"<fresh-venv>": str(venv), "<wheel>": str(wheel.resolve())}, runner)
        failed = next((result for result in results if result.status != "pass"), None)
        if failed is not None:
            return _wheel_gate_failure(failed.reason, failed.detail, failed.command)
        evidence = {"wheel": str(wheel.resolve()), "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "module_path": results[2].detail, "venv": str(venv)}
        return GateResult("wheel-abi-contract", "pass", "OK", "release wheel ABI contract verified", results[-1].command, evidence=evidence)
