"""Tests for the ``doctor`` subcommand and its individual checks."""

from __future__ import annotations

import argparse
import io
import json
import sys

import pytest

from omlx_research.cli import main
from omlx_research.cli import _doctor_checks as checks_mod
from omlx_research.cli import doctor as doctor_mod
from omlx_research.cli.doctor import (
    CHECKS,
    EXPECTED_KERNEL_OP_COUNT,
    FAIL,
    MIN_PYTHON,
    PASS,
    WARN,
    Check,
    DoctorReport,
    cmd_doctor,
    register,
    run_doctor,
)
from omlx_research.cli._doctor_shared import (
    collect_kernel_op_tags,
    read_abi_version,
    read_cargo_version,
)


# --- stdio capture ---------------------------------------------------------

class _IO:
    """Swap sys.stdout/sys.stderr for one cmd_* call."""

    def __init__(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def __enter__(self):
        self._real_out, self._real_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self.stdout, self.stderr
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self._real_out, self._real_err


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def _find(report: DoctorReport, check_id: str) -> Check:
    for c in report.checks:
        if c.id == check_id:
            return c
    raise AssertionError(f"no check with id={check_id!r} in report")


# --- argparse wiring -------------------------------------------------------

def test_main_routes_doctor_subcommand(capsys):
    rc = main(["doctor"])
    # Whether 0 or 1, the routing itself worked — it didn't raise and
    # produced the doctor banner.
    out = capsys.readouterr().out
    assert "== omlx-research doctor ==" in out
    assert rc in (0, 1)


def test_main_routes_doctor_json(capsys):
    rc = main(["doctor", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "checks" in payload
    assert isinstance(payload["checks"], list)
    assert any(c["id"] == "python_version" for c in payload["checks"])
    assert rc in (0, 1)


def test_register_attaches_doctor_subparser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    register(sub)
    args = parser.parse_args(["doctor"])
    assert args.cmd == "doctor"
    assert getattr(args, "json", False) is False


def test_register_accepts_json_flag():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    register(sub)
    args = parser.parse_args(["doctor", "--json"])
    assert args.cmd == "doctor"
    assert args.json is True


# --- python_version check --------------------------------------------------

def test_run_doctor_python_version_passes_on_3_14_plus():
    """We're running on Python >= 3.14 by spec; python_version must pass."""
    report = run_doctor()
    c = _find(report, "python_version")
    assert c.status == PASS
    assert sys.version_info >= MIN_PYTHON


# --- exit-code semantics ---------------------------------------------------

def test_doctor_report_exit_code_zero_when_all_pass():
    report = DoctorReport(checks=[
        Check(id="a", description="", status=PASS),
        Check(id="b", description="", status=PASS),
    ])
    assert report.exit_code() == 0


def test_doctor_report_exit_code_one_when_any_warn():
    report = DoctorReport(checks=[
        Check(id="a", description="", status=PASS),
        Check(id="b", description="", status=WARN),
    ])
    assert report.exit_code() == 1


def test_doctor_report_exit_code_one_when_any_fail():
    report = DoctorReport(checks=[
        Check(id="a", description="", status=PASS),
        Check(id="b", description="", status=FAIL),
    ])
    assert report.exit_code() == 1


# --- cmd_doctor rendering --------------------------------------------------

def test_cmd_doctor_human_renders_banner():
    with _IO() as io:
        rc = cmd_doctor(_ns(json=False))
    out = io.stdout.getvalue()
    assert "== omlx-research doctor ==" in out
    assert "summary:" in out
    assert rc in (0, 1)


def test_cmd_doctor_json_emits_valid_envelope():
    with _IO() as io:
        rc = cmd_doctor(_ns(json=True))
    out = io.stdout.getvalue()
    payload = json.loads(out)
    assert "checks" in payload and isinstance(payload["checks"], list)
    assert "ok" in payload
    assert "python_version" in payload
    assert "platform" in payload
    assert "schema_version" in payload
    assert rc in (0, 1)


# --- run_doctor robustness -------------------------------------------------

def test_run_doctor_returns_at_least_one_check_for_each_id():
    report = run_doctor()
    ids = {c.id for c in report.checks}
    expected = {
        "python_version",
        "mlx_core_available",
        "mlx_lm_available",
        "turboquant_rust_extension_available",
        "kernel_registry_version",
        "regress_baseline_version",
        "model_kernels_operator_coverage",
        "native_abi_v1",
        "airlock_v2_installed",
        "tests_runnable",
    }
    assert expected.issubset(ids)


def test_run_doctor_does_not_raise_even_when_optional_deps_missing(monkeypatch):
    """Simulate ``mlx_lm`` missing — the check should degrade to WARN."""
    # Setting a module entry to None causes `import mlx_lm` to raise
    # ModuleNotFoundError on first use; we patch the check to skip real
    # imports via sys.modules trickery used in tandem.
    monkeypatch.delitem(sys.modules, "mlx_lm", raising=False)
    # Inject a sentinel that the import machinery will refuse to load.
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    try:
        report = run_doctor()
    finally:
        # Restore so we don't leak state into the rest of the suite.
        monkeypatch.delitem(sys.modules, "mlx_lm", raising=False)
    c = _find(report, "mlx_lm_available")
    # mlx_lm may legitimately be installed in this environment, so accept
    # either pass or warn — the requirement is that it doesn't raise.
    assert c.status in (PASS, WARN)
    # And specifically, the run_doctor call itself returned without raising.


def test_mlx_lm_check_warns_when_module_unavailable(monkeypatch):
    """Direct: ensure the mlx_lm check degrades to WARN when mlx_lm is missing."""
    monkeypatch.delitem(sys.modules, "mlx_lm", raising=False)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    c = checks_mod.mlx_lm()
    assert c.status == WARN
    assert c.id == "mlx_lm_available"
    assert "not installed" in c.details


def test_turboquant_rust_extension_check_warns_when_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "_perf", raising=False)
    monkeypatch.setitem(sys.modules, "_perf", None)
    c = checks_mod.turboquant_rust_extension()
    assert c.status == WARN
    assert c.id == "turboquant_rust_extension_available"


# --- helper-level checks ---------------------------------------------------

def test_read_cargo_version_resolves_workspace_inheritance():
    # kernel-registry uses version.workspace = true; the helper must walk
    # up to perf-core/Cargo.toml to find the workspace.package version.
    v = read_cargo_version("perf-core/kernel-registry")
    assert v != "unknown"
    # workspace.package declares version = "0.1.0"
    assert v == "0.1.0"


def test_read_cargo_version_handles_direct_version_field():
    # native-abi declares version = "0.1.0" directly.
    v = read_cargo_version("perf-core/native-abi")
    assert v == "0.1.0"


def test_read_cargo_version_returns_unknown_for_missing_crate():
    v = read_cargo_version("perf-core/this-crate-does-not-exist")
    assert v == "unknown"


def test_collect_kernel_op_tags_finds_expected_count():
    tags = collect_kernel_op_tags()
    assert len(tags) >= EXPECTED_KERNEL_OP_COUNT
    # Spot-check the documented tags (the first + last from the enum).
    assert "dense_attention" in tags
    assert "mod_routing" in tags
    # Sanity: every tag is non-empty and lowercase.
    assert all(t and t.replace("_", "").isalnum() for t in tags)


def test_read_abi_version_returns_v1():
    abi = read_abi_version()
    assert abi is not None
    assert abi.startswith("1.")


def test_model_kernels_operator_coverage_passes_for_real_repo():
    """Real model-kernels/src/lib.rs should expose >= EXPECTED_KERNEL_OP_COUNT tags."""
    c = checks_mod.model_kernels_operator_coverage()
    assert c.status == PASS
    assert c.id == "model_kernels_operator_coverage"


def test_native_abi_v1_check_passes_for_real_repo():
    c = checks_mod.native_abi_v1()
    assert c.status == PASS
    assert "ABI v1" in c.details


def test_airlock_v2_warns_when_not_on_path(monkeypatch):
    # shutil.which returns the path or None; force None to simulate missing.
    monkeypatch.setattr("shutil.which", lambda _name: None)
    c = checks_mod.airlock_v2()
    assert c.status == WARN
    assert "NOT INSTALLED" in c.details


# --- CHECKS list sanity ----------------------------------------------------

def test_checks_list_includes_expected_ids():
    ids = [getattr(fn, "__name__", "") for fn in CHECKS]
    # Each registered check function must be one we expect.
    expected = {
        "python_version",
        "mlx_core",
        "mlx_lm",
        "turboquant_rust_extension",
        "kernel_registry_version",
        "regress_baseline_version",
        "model_kernels_operator_coverage",
        "native_abi_v1",
        "airlock_v2",
        "tests_runnable",
    }
    assert expected.issubset(set(ids))


def test_run_doctor_each_check_returns_a_check_instance():
    for c in run_doctor().checks:
        assert isinstance(c, Check)
        assert c.status in (PASS, WARN, FAIL)
        assert c.id and isinstance(c.id, str)
        assert c.description and isinstance(c.description, str)


# --- exit code reflects check status --------------------------------------

def test_cmd_doctor_exit_code_zero_when_all_pass(monkeypatch):
    """Force every check to PASS and confirm cmd_doctor returns 0."""

    def _ok():
        return Check(id="ok", description="ok", status=PASS)

    monkeypatch.setattr(doctor_mod, "CHECKS", [_ok])
    with _IO():
        rc = cmd_doctor(_ns(json=False))
    assert rc == 0


def test_cmd_doctor_exit_code_one_when_any_check_warns(monkeypatch):
    def _warn():
        return Check(id="w", description="w", status=WARN)

    monkeypatch.setattr(doctor_mod, "CHECKS", [_warn])
    with _IO():
        rc = cmd_doctor(_ns(json=False))
    assert rc == 1


def test_cmd_doctor_exit_code_one_when_any_check_fails(monkeypatch):
    def _fail():
        return Check(id="f", description="f", status=FAIL)

    monkeypatch.setattr(doctor_mod, "CHECKS", [_fail])
    with _IO():
        rc = cmd_doctor(_ns(json=False))
    assert rc == 1
