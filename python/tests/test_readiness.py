"""Contract tests for deterministic readiness orchestration."""

from __future__ import annotations

import importlib.util
import hashlib
from contextlib import contextmanager
from pathlib import Path
from subprocess import CompletedProcess


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "phenotype_readiness", ROOT / "scripts" / "readiness_check.py"
)
assert SPEC and SPEC.loader
readiness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readiness)


def test_readiness_entrypoint_stays_within_size_limit() -> None:
    assert len((ROOT / "scripts" / "readiness_check.py").read_text().splitlines()) <= 500


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_cargo_resolution_precedence_prefers_executable_env(tmp_path: Path) -> None:
    configured = _executable(tmp_path / "configured" / "cargo")
    path_cargo = _executable(tmp_path / "path" / "cargo")
    rustup_home = tmp_path / "rustup"
    _executable(rustup_home / "toolchains" / "nightly-aarch64-apple-darwin" / "bin" / "cargo")

    resolved = readiness.resolve_cargo(
        env={"CARGO": str(configured)},
        which=lambda _: str(path_cargo),
        rustup_home=rustup_home,
        host_triple="aarch64-apple-darwin",
    )

    assert resolved == str(configured)


def test_cargo_resolution_falls_back_from_path_to_host_rustup(tmp_path: Path) -> None:
    rustup_home = tmp_path / "rustup"
    expected = _executable(
        rustup_home / "toolchains" / "nightly-aarch64-apple-darwin" / "bin" / "cargo"
    )
    _executable(rustup_home / "toolchains" / "stable-x86_64-unknown-linux-gnu" / "bin" / "cargo")

    resolved = readiness.resolve_cargo(
        env={"CARGO": str(tmp_path / "not-executable")},
        which=lambda _: None,
        rustup_home=rustup_home,
        host_triple="aarch64-apple-darwin",
    )

    assert resolved == str(expected)


def test_cargo_resolution_uses_path_before_rustup(tmp_path: Path) -> None:
    path_cargo = _executable(tmp_path / "path" / "cargo")
    rustup_home = tmp_path / "rustup"
    _executable(rustup_home / "toolchains" / "nightly-aarch64-apple-darwin" / "bin" / "cargo")

    resolved = readiness.resolve_cargo(
        env={},
        which=lambda _: str(path_cargo),
        rustup_home=rustup_home,
        host_triple="aarch64-apple-darwin",
    )

    assert resolved == str(path_cargo)


def test_native_runner_exposes_toolchain_sibling_rustc(
    tmp_path: Path, monkeypatch
) -> None:
    cargo = tmp_path / "toolchain" / "bin" / "cargo"
    cargo.parent.mkdir(parents=True)
    cargo.write_text("#!/bin/sh\nrustc --version\n")
    cargo.chmod(0o755)
    rustc = cargo.with_name("rustc")
    rustc.write_text("#!/bin/sh\necho rustc-sibling\n")
    rustc.chmod(0o755)
    monkeypatch.setenv("PATH", "")

    completed = readiness._subprocess_runner([str(cargo)], tmp_path)

    assert completed.returncode == 0
    assert completed.stdout.strip() == "rustc-sibling"


def test_missing_cargo_returns_structured_failure(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "checkout"
    (root / "perf-core").mkdir(parents=True)
    monkeypatch.setattr(readiness, "resolve_cargo", lambda: None)

    report = readiness.run_readiness(root)

    assert report.ok is False
    assert [result.reason for result in report.results] == [
        "COMMAND_NOT_FOUND",
        "COMMAND_NOT_FOUND",
    ]
    assert [result.command for result in report.results] == [
        ["cargo", "check", "--workspace"],
        ["cargo", "test", "--workspace"],
    ]


def test_cargo_validation_is_not_skipped_when_target_exists(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    (root / "perf-core" / "target").mkdir(parents=True)
    calls: list[tuple[tuple[str, ...], Path]] = []

    def runner(command, cwd, env=None):
        calls.append((tuple(command), cwd))
        return CompletedProcess(command, 0, stdout="ok", stderr="")

    report = readiness.run_readiness(
        root, runner=runner, cargo="cargo", include_python=False
    )

    assert [result.gate for result in report.results] == [
        "cargo-check",
        "cargo-test",
        "wheel-abi-contract",
    ]
    assert report.results[-1].status == "not-run"
    assert calls == [
        (("cargo", "check", "--workspace"), root / "perf-core"),
        (("cargo", "test", "--workspace"), root / "perf-core"),
    ]
    assert report.ok is True


def test_fast_readiness_reports_wheel_artifact_as_not_verified(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    (root / "perf-core").mkdir(parents=True)

    def runner(command, cwd, env=None):
        return CompletedProcess(command, 0, stdout="ok", stderr="")

    report = readiness.run_readiness(
        root, runner=runner, cargo="cargo", include_python=False
    )

    assert report.ok is True
    assert report.fully_verified is False
    assert report.as_dict()["status"] == "partial"
    assert [(result.gate, result.status, result.reason) for result in report.results] == [
        ("cargo-check", "pass", "OK"),
        ("cargo-test", "pass", "OK"),
        ("wheel-abi-contract", "not-run", "FULL_MODE_REQUIRED"),
    ]


def test_full_readiness_runs_wheel_artifact_contract(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "checkout"
    (root / "perf-core").mkdir(parents=True)
    invoked = []

    def runner(command, cwd, env=None):
        return CompletedProcess(command, 0, stdout="ok", stderr="")

    def wheel_gate(received_root, *, runner, python_executable, maturin_executable):
        invoked.append((received_root, runner, python_executable, maturin_executable))
        return readiness.GateResult(
            "wheel-abi-contract",
            "pass",
            "OK",
            "wheel verified",
            ["maturin", "build"],
            evidence={"wheel_sha256": "a" * 64},
        )

    monkeypatch.setattr(readiness, "run_wheel_abi_contract_gate", wheel_gate)
    report = readiness.run_readiness(
        root, runner=runner, cargo="cargo", include_python=False, include_wheel_abi=True
    )

    assert report.ok is True
    assert report.fully_verified is True
    assert report.as_dict()["status"] == "pass"
    assert [result.status for result in report.results] == ["pass", "pass", "pass"]
    assert invoked and invoked[0][0] == root.resolve()


def test_failed_command_has_stable_structured_result(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    (root / "perf-core").mkdir(parents=True)

    def runner(command, cwd, env=None):
        return CompletedProcess(command, 7, stdout="", stderr="compiler failed\n")

    report = readiness.run_readiness(
        root, runner=runner, cargo="cargo", include_python=False
    )
    result = report.results[0]

    assert result.as_dict() == {
        "gate": "cargo-check",
        "status": "fail",
        "reason": "COMMAND_FAILED",
        "detail": "compiler failed",
        "command": ["cargo", "check", "--workspace"],
    }
    assert report.ok is False


def test_shell_entrypoint_is_a_thin_python_delegate() -> None:
    wrapper = (ROOT / "scripts" / "phenotype-omlx-ready").read_text()

    assert 'COLLECTION_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"' in wrapper
    assert '"${SCRIPT_DIR}/polyrepo_boundary.py" "$COLLECTION_ROOT"' in wrapper
    assert 'PYTHON_BIN="${PYTHON_BIN:-python3}"' in wrapper
    assert 'exec "${PYTHON_BIN}" "${SCRIPT_DIR}/readiness_check.py" "$@"' in wrapper
    assert "target" not in wrapper


def test_source_gate_enumerates_only_source_safe_tests(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    (root / "python" / "tests").mkdir(parents=True)
    calls = []

    def runner(command, cwd, env=None):
        calls.append((list(command), cwd, env))
        return CompletedProcess(command, 0, stdout="2 passed", stderr="")

    result = readiness.run_source_python_gate(root, runner, "python3")

    assert result.status == "pass"
    command, cwd, env = calls[0]
    assert command == [
        "python3",
        "-m",
        "pytest",
        str(root / "python/tests/test_mlx_backend.py"),
        str(root / "python/tests/test_readiness.py"),
        "-q",
    ]
    assert "test_perf_extension.py" not in command
    assert cwd == root
    assert env["PYTHONPATH"] == str(root / "python")


def test_wheel_abi_gate_plan_excludes_source_and_locks_build(tmp_path: Path) -> None:
    root = tmp_path / "checkout"

    plan = readiness.wheel_abi_gate_plan(root, "python3", "maturin")

    build, environment, install, identity, abi_test, console = plan
    assert build.cwd == root / "python"
    assert build.command == ["maturin", "build", "--release", "--locked"]
    assert environment.command == [
        "python3",
        "-m",
        "venv",
        "--system-site-packages",
        "<fresh-venv>",
    ]
    assert install.command == [
        "<fresh-venv>/bin/python",
        "-m",
        "pip",
        "install",
        "--no-deps",
        "<wheel>",
    ]
    assert identity.cwd == Path("/")
    assert identity.env["PYTHONPATH"] == ""
    assert identity.command == [
        "<fresh-venv>/bin/python",
        "-I",
        "-c",
        readiness.WHEEL_IDENTITY_SCRIPT,
    ]
    assert '_perf.__name__ == "omlx_research._perf"' in readiness.WHEEL_IDENTITY_SCRIPT
    assert 'find_spec("_perf") is None' in readiness.WHEEL_IDENTITY_SCRIPT
    assert "module_path.is_relative_to(prefix)" in readiness.WHEEL_IDENTITY_SCRIPT
    assert abi_test.cwd == Path("/")
    assert abi_test.env["PYTHONPATH"] == ""
    assert abi_test.command == [
        "<fresh-venv>/bin/python",
        "-I",
        "-m",
        "pytest",
        str(root / "python/tests/test_perf_extension.py"),
        "-q",
    ]
    assert console.cwd == Path("/")
    assert console.env["PYTHONPATH"] == ""
    assert console.command == ["<fresh-venv>/bin/omlx-research", "--help"]


def test_execute_planned_commands_substitutes_in_order_and_stops_on_failure(
    tmp_path: Path,
) -> None:
    calls = []
    plan = (
        readiness.PlannedCommand(
            ["<tool>", "--value=<token>"],
            Path("<workspace>") / "first",
            {"PLAN_TOKEN": "<token>"},
        ),
        readiness.PlannedCommand(
            ["<tool>", "fail"],
            Path("<workspace>") / "second",
            {"PLAN_ROOT": "<workspace>"},
        ),
        readiness.PlannedCommand(["must-not-run"], Path("/"), {}),
    )

    def runner(command, cwd, env=None):
        calls.append((list(command), cwd, dict(env or {})))
        return CompletedProcess(
            command,
            0 if len(calls) == 1 else 7,
            stdout="first completed" if len(calls) == 1 else "",
            stderr="" if len(calls) == 1 else "second failed",
        )

    results = readiness.execute_planned_commands(
        plan,
        {"<tool>": "tool", "<token>": "resolved", "<workspace>": str(tmp_path)},
        runner,
    )

    assert [result.status for result in results] == ["pass", "fail"]
    assert [result.reason for result in results] == ["OK", "COMMAND_FAILED"]
    assert calls == [
        (
            ["tool", "--value=resolved"],
            tmp_path / "first",
            {"PLAN_TOKEN": "resolved"},
        ),
        (
            ["tool", "fail"],
            tmp_path / "second",
            {"PLAN_ROOT": str(tmp_path)},
        ),
    ]


def test_execute_planned_commands_records_sanitized_evidence_and_stops_on_exception(
    tmp_path: Path,
) -> None:
    calls = []
    plan = (
        readiness.PlannedCommand(
            ["<tool>", "first"],
            Path("<workspace>") / "first",
            {"PLANNED_TOKEN": "<token>"},
        ),
        readiness.PlannedCommand(["must-not-run"], Path("/"), {}),
    )

    def runner(command, cwd, env=None):
        calls.append((list(command), cwd, dict(env or {})))
        raise FileNotFoundError("tool missing")

    results = readiness.execute_planned_commands(
        plan,
        {"<tool>": "tool", "<token>": "resolved", "<workspace>": str(tmp_path)},
        runner,
    )

    assert [(result.status, result.reason) for result in results] == [
        ("fail", "COMMAND_NOT_FOUND")
    ]
    assert results[0].cwd == str(tmp_path / "first")
    assert results[0].env == {"PLANNED_TOKEN": "resolved"}
    assert calls == [
        (["tool", "first"], tmp_path / "first", {"PLANNED_TOKEN": "resolved"})
    ]


def test_execute_planned_commands_stops_before_later_command_on_oserror() -> None:
    calls = []
    plan = (
        readiness.PlannedCommand(["first"], Path("/"), {}),
        readiness.PlannedCommand(["must-not-run"], Path("/"), {}),
    )

    def runner(command, cwd, env=None):
        calls.append(list(command))
        raise OSError("runner unavailable")

    results = readiness.execute_planned_commands(plan, {}, runner)

    assert [(result.status, result.reason) for result in results] == [
        ("fail", "COMMAND_ERROR")
    ]
    assert calls == [["first"]]


def test_subprocess_runner_uses_only_planned_environment(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}
    monkeypatch.setenv("READINESS_SENTINEL_SECRET", "must-not-leak")

    def fake_run(*args, **kwargs):
        captured.update(kwargs["env"])
        return CompletedProcess(args[0], 0, stdout="ok", stderr="")

    monkeypatch.setattr(readiness.readiness_commands.subprocess, "run", fake_run)

    completed = readiness._subprocess_runner(["tool"], tmp_path, {"PLANNED": "yes"})

    assert completed.returncode == 0
    assert captured == {"PLANNED": "yes"}


def test_wheel_abi_contract_runs_verified_plan_with_isolated_artifact_workspace(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    wheel_dir = root / "python" / "ffi" / "target" / "wheels"
    artifact_workspace = tmp_path / "artifacts"
    calls = []

    @contextmanager
    def workspace_factory():
        artifact_workspace.mkdir()
        yield artifact_workspace

    def runner(command, cwd, env=None):
        calls.append((list(command), cwd, dict(env or {})))
        if command[1:3] == ["build", "--release"]:
            wheel_dir.mkdir(parents=True)
            (wheel_dir / "omlx_research-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
        stdout = "/fresh/site-packages/omlx_research/_perf.so" if "-c" in command else "ok"
        return CompletedProcess(command, 0, stdout=stdout, stderr="")

    result = readiness.run_wheel_abi_contract_gate(
        root,
        runner=runner,
        python_executable="python3",
        maturin_executable="maturin",
        workspace_factory=workspace_factory,
    )

    assert result.gate == "wheel-abi-contract"
    assert result.status == "pass"
    assert result.evidence["wheel_sha256"] == hashlib.sha256(b"wheel").hexdigest()
    assert result.evidence["module_path"] == "/fresh/site-packages/omlx_research/_perf.so"
    assert len(calls) == 6
    assert calls[0][0] == ["maturin", "build", "--release", "--locked"]
    assert calls[1][0] == [
        "python3",
        "-m",
        "venv",
        "--system-site-packages",
        str(artifact_workspace / "venv"),
    ]
    assert calls[-1][0] == [str(artifact_workspace / "venv" / "bin" / "omlx-research"), "--help"]
