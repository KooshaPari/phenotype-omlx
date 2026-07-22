"""Command planning, execution, and result types for readiness gates."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Callable, Mapping, Sequence

try:
    from readiness_toolchain import host_triple as _host_triple, is_executable as _is_executable
except ModuleNotFoundError:
    from scripts.readiness_toolchain import host_triple as _host_triple, is_executable as _is_executable


CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str] | None], subprocess.CompletedProcess[str]]

WHEEL_IDENTITY_SCRIPT = (
    "import importlib.util, pathlib, sys; "
    "from omlx_research import _perf; "
    "module_path = pathlib.Path(_perf.__file__).resolve(); "
    "prefix = pathlib.Path(sys.prefix).resolve(); "
    'assert _perf.__name__ == "omlx_research._perf"; '
    'assert importlib.util.find_spec("_perf") is None; '
    "assert module_path.is_relative_to(prefix); "
    "print(module_path)"
)


class PlannedCommand:
    """One deterministic command in a future isolated wheel ABI gate."""

    def __init__(self, command: Sequence[str], cwd: Path, env: Mapping[str, str]) -> None:
        self.command = list(command)
        self.cwd = cwd
        self.env = dict(env)


class GateResult:
    """Stable result for one independently actionable readiness gate."""

    def __init__(self, gate: str, status: str, reason: str, detail: str, command: Sequence[str], cwd: Path | None = None, env: Mapping[str, str] | None = None, evidence: Mapping[str, str] | None = None) -> None:
        self.gate, self.status, self.reason, self.detail = gate, status, reason, detail
        self.command, self.cwd, self.env = list(command), str(cwd) if cwd else None, dict(env or {})
        self._has_execution_evidence = env is not None
        self.evidence = dict(evidence or {})

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"gate": self.gate, "status": self.status, "reason": self.reason, "detail": self.detail, "command": self.command}
        if self._has_execution_evidence:
            payload.update(cwd=self.cwd, env=self.env)
        if self.evidence:
            payload["evidence"] = self.evidence
        return payload


class ReadinessReport:
    """Ordered collection of readiness gate results."""

    def __init__(self, root: Path, results: list[GateResult]) -> None:
        self.root, self.results = root, results

    @property
    def ok(self) -> bool:
        return all(result.status != "fail" for result in self.results)

    @property
    def fully_verified(self) -> bool:
        return all(result.status == "pass" for result in self.results)

    def as_dict(self) -> dict[str, object]:
        return {"schema_version": 1, "root": str(self.root), "status": "fail" if not self.ok else "pass" if self.fully_verified else "partial", "results": [result.as_dict() for result in self.results]}


def wheel_abi_gate_plan(root: Path, python_executable: str, maturin_executable: str) -> tuple[PlannedCommand, ...]:
    """Describe, without running, the source-isolated release-wheel ABI gate."""
    resolved_root = root.expanduser().resolve()
    python_root, venv_python = resolved_root / "python", "<fresh-venv>/bin/python"
    source_excluded = {"PYTHONPATH": ""}
    return (
        PlannedCommand([maturin_executable, "build", "--release", "--locked"], python_root, {}),
        PlannedCommand([python_executable, "-m", "venv", "--system-site-packages", "<fresh-venv>"], resolved_root, {}),
        PlannedCommand([venv_python, "-m", "pip", "install", "--no-deps", "<wheel>"], resolved_root, {}),
        PlannedCommand([venv_python, "-I", "-c", WHEEL_IDENTITY_SCRIPT], Path("/"), source_excluded),
        PlannedCommand([venv_python, "-I", "-m", "pytest", str(python_root / "tests" / "test_perf_extension.py"), "-q"], Path("/"), source_excluded),
        PlannedCommand(["<fresh-venv>/bin/omlx-research", "--help"], Path("/"), source_excluded),
    )


def resolve_cargo(*, env: Mapping[str, str] | None = None, which: Callable[[str], str | None] = shutil.which, rustup_home: Path | None = None, host_triple: str | None = None) -> str | None:
    """Resolve Cargo deterministically: explicit env, PATH, then host Rustup tools."""
    environment = os.environ if env is None else env
    configured = environment.get("CARGO")
    if configured and _is_executable(Path(configured).expanduser()):
        return str(Path(configured).expanduser())
    on_path = which("cargo")
    if on_path and _is_executable(Path(on_path)):
        return on_path
    triple = _host_triple() if host_triple is None else host_triple
    if triple is None:
        return None
    home = rustup_home or Path(environment.get("RUSTUP_HOME", "~/.rustup")).expanduser()
    candidates = [path for path in (home / "toolchains").glob(f"*{triple}/bin/cargo") if _is_executable(path)]
    if not candidates:
        return None

    def stable_version(path: Path) -> tuple[int, int, int] | None:
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)-", path.parent.parent.name)
        return tuple(map(int, match.groups())) if match else None

    stable = [(stable_version(path), path) for path in candidates]
    versioned = [(version, path) for version, path in stable if version is not None]
    selected = max(versioned, key=lambda item: item[0])[1] if versioned else sorted(candidates)[-1]
    return str(selected)


def _subprocess_runner(command: Sequence[str], cwd: Path, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy() if env is None else dict(env)
    executable = Path(command[0]).expanduser()
    if executable.parent != Path("."):
        environment["PATH"] = os.pathsep.join(part for part in (str(executable.parent), environment.get("PATH", "")) if part)
    return subprocess.run(list(command), cwd=cwd, capture_output=True, text=True, check=False, env=environment)


def _run_command_gate(gate: str, command: Sequence[str], cwd: Path, runner: CommandRunner, env: Mapping[str, str] | None = None, evidence_env: Mapping[str, str] | None = None) -> GateResult:
    try:
        completed = runner(command, cwd, env)
    except FileNotFoundError as error:
        return GateResult(gate, "fail", "COMMAND_NOT_FOUND", str(error), command, cwd, evidence_env)
    except OSError as error:
        return GateResult(gate, "fail", "COMMAND_ERROR", str(error), command, cwd, evidence_env)
    output = (completed.stderr or completed.stdout or "").strip()
    return GateResult(gate, "fail" if completed.returncode else "pass", "COMMAND_FAILED" if completed.returncode else "OK", output or "completed", command, cwd, evidence_env)


def execute_planned_commands(plan: Sequence[PlannedCommand], substitutions: Mapping[str, str], runner: CommandRunner) -> list[GateResult]:
    """Execute a substituted command plan, retaining ordered failure evidence."""
    def substitute(value: str) -> str:
        for placeholder, replacement in substitutions.items():
            value = value.replace(placeholder, replacement)
        return value
    results: list[GateResult] = []
    for index, planned in enumerate(plan, start=1):
        command = [substitute(argument) for argument in planned.command]
        cwd, environment = Path(substitute(str(planned.cwd))), {key: substitute(value) for key, value in planned.env.items()}
        result = _run_command_gate(f"planned-command-{index}", command, cwd, runner, environment, environment)
        results.append(result)
        if result.status != "pass":
            break
    return results
