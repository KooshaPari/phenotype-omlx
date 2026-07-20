"""Tests for the doctor checks added 2026-07-19.

Covers both the turn-4 batch (omlx_research version, NIAH benchmark,
eval-harness, regress-baseline dispatch envelope) and the turn-5 batch
(NIAH regression baseline, dispatch script probes). Mirrors the
patterns in :mod:`omlx_research.cli.tests.test_doctor` but isolates
the new check coverage into its own module so ``test_doctor.py``
stays under the 500L hard cap. Each test exercises either a
monkey-patched failure branch (for ``unittest.mock.patch``-style
coverage of missing modules) or a real-repo smoke test that runs
against the live repository.
"""

from __future__ import annotations

import json
import os
import sys

from omlx_research.cli import _doctor_checks as checks_mod
from omlx_research.cli import _doctor_extra_eval as eval_extra
from omlx_research.cli import _doctor_extra_kernel as kernel_extra
from omlx_research.cli import _doctor_extra_niah as niah_extra
from omlx_research.cli import _doctor_turn5_checks as turn5
from omlx_research.cli.doctor import (
    FAIL,
    PASS,
    WARN,
)


# --- omlx_research_version ------------------------------------------------


def test_omlx_research_version_passes_for_installed_package():
    """The version check must PASS and surface omlx_research.__version__."""
    c = checks_mod.omlx_research_version()
    assert c.status == PASS
    assert c.id == "omlx_research_version"
    import omlx_research  # noqa: F401
    assert omlx_research.__version__ in c.details


def test_omlx_research_version_warns_when_version_attr_missing(monkeypatch):
    """An importable omlx_research without __version__ degrades to WARN."""
    import omlx_research

    monkeypatch.delattr(omlx_research, "__version__", raising=False)
    c = checks_mod.omlx_research_version()
    assert c.status == WARN
    assert "__version__ is missing" in c.details


# --- niah_benchmark_present -----------------------------------------------


def test_niah_benchmark_present_passes_when_script_exists(monkeypatch):
    """With a stubbed script that exits 0 on --help, the check PASSES."""
    fake_script = "/tmp/__fake_niah__.py"

    def _fake_find():
        return fake_script

    class _FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(niah_extra, "_find_niah_benchmark", _fake_find)
    monkeypatch.setattr(
        niah_extra.subprocess, "run",
        lambda *a, **kw: _FakeProc(),
    )
    c = checks_mod.niah_benchmark_present()
    assert c.status == PASS
    assert c.id == "niah_benchmark_present"
    # The path is reported via os.path.relpath; just confirm the
    # marker text "--help exits 0" survives, which is the new bit.
    assert "--help exits 0" in c.details
    assert "NIAH benchmark executable" in c.details


def test_niah_benchmark_present_warns_when_script_missing(monkeypatch):
    """Missing script -> WARN with the documented needle-in-haystack message."""
    monkeypatch.setattr(niah_extra, "_find_niah_benchmark", lambda: None)
    c = checks_mod.niah_benchmark_present()
    assert c.status == WARN
    assert "NIAH benchmark absent" in c.details


def test_niah_benchmark_present_warns_when_help_fails(monkeypatch):
    """Present script that fails --help -> WARN."""
    class _FakeProc:
        returncode = 2
        stdout = ""
        stderr = "usage: bad args"

    monkeypatch.setattr(
        niah_extra, "_find_niah_benchmark",
        lambda: "/tmp/__fake_niah__.py",
    )
    monkeypatch.setattr(
        niah_extra.subprocess, "run",
        lambda *a, **kw: _FakeProc(),
    )
    c = checks_mod.niah_benchmark_present()
    assert c.status == WARN
    assert "exited 2" in c.details


# --- eval_harness_subcommand_runnable -------------------------------------


def test_eval_harness_subcommand_warns_when_python_missing_but_rust_present(monkeypatch):
    """Rust crate present but `eval` subcommand missing -> WARN.

    The eval-harness is a pure-Rust crate consumed via the
    kernel-registry; the canonical Python entry point is the CLI
    ``eval`` subcommand. When the subcommand is absent the user can
    still drive the harness via the kernel-registry, so we degrade to
    WARN (not FAIL).
    """
    monkeypatch.setattr(
        eval_extra, "_cli_has_eval_subcommand",
        lambda: False,
    )
    monkeypatch.setattr(
        eval_extra, "_eval_harness_rust_crate",
        lambda: (True, "perf-core/eval-harness/"),
    )
    monkeypatch.setattr(eval_extra, "_list_eval_harness_tests", lambda: [])
    c = checks_mod.eval_harness_subcommand_runnable()
    assert c.status == WARN
    assert c.id == "eval_harness_subcommand_runnable"
    assert "Rust crate" in c.details
    assert "perf-core/eval-harness/" in c.details
    assert "subcommand" in c.details


def test_eval_harness_subcommand_warns_when_both_missing(monkeypatch):
    """No `eval` subcommand AND Rust crate absent -> WARN.

    Both surfaces gone is a serious gap, but we surface it as WARN
    rather than FAIL because the eval-harness is consumed by the
    kernel-registry, not the CLI directly — the CLI can still serve
    decode/inference paths without it.
    """
    monkeypatch.setattr(
        eval_extra, "_cli_has_eval_subcommand",
        lambda: False,
    )
    monkeypatch.setattr(
        eval_extra, "_eval_harness_rust_crate",
        lambda: (False, "perf-core/eval-harness/Cargo.toml not found"),
    )
    c = checks_mod.eval_harness_subcommand_runnable()
    assert c.status == WARN
    assert "Both surfaces are absent" in c.details


def test_eval_harness_subcommand_passes_when_subcommand_registered(monkeypatch):
    """`eval` subcommand registered -> PASS.

    Once the CLI subcommand is registered, the user has an end-to-end
    Python entry point into the Rust eval-harness crate. The check no
    longer requires ``omlx_research.eval`` to be a separate Python
    module — the CLI subcommand is the canonical Python surface.
    """
    monkeypatch.setattr(eval_extra, "_cli_has_eval_subcommand", lambda: True)
    monkeypatch.setattr(eval_extra, "_list_eval_harness_tests", lambda: ["test_eval_subcommand.py"])
    c = checks_mod.eval_harness_subcommand_runnable()
    assert c.status == PASS
    assert c.id == "eval_harness_subcommand_runnable"
    assert "eval` subcommand registered" in c.details
    assert "test_eval_subcommand.py" in c.details


# --- regress_baseline_dispatch_envelope ----------------------------------


def test_regress_baseline_dispatch_envelope_warns_when_extension_missing(tmp_path, monkeypatch):
    """Missing regress_baseline module AND no niah_results.json -> WARN.

    The check now treats the populated niah_results.json as the
    canonical envelope reference, so the test must isolate
    project_root() to a tmp_path that does not contain either
    signal. That way the supplementary regress_baseline probe is
    exercised but cannot promote the check to PASS on its own.
    """
    monkeypatch.delitem(sys.modules, "regress_baseline", raising=False)
    monkeypatch.setitem(sys.modules, "regress_baseline", None)
    monkeypatch.setattr(niah_extra, "project_root", lambda: str(tmp_path))
    c = checks_mod.regress_baseline_dispatch_envelope()
    assert c.status == WARN
    assert c.id == "regress_baseline_dispatch_envelope"
    assert "not built" in c.details


def test_regress_baseline_dispatch_envelope_passes_with_stub_module(monkeypatch):
    """A stub regress_baseline exposing dispatch_budget() -> PASS.

    The check now PASSes on the populated niah_results.json alone
    (the canonical envelope reference), so the test only needs to
    confirm the supplementary dispatch_budget() probe surfaces its
    value in the details string when both signals are healthy.
    """

    class _Stub:
        # Stub the 3-arg form (m, n, k) — the check uses this when the
        # module does not expose a ShapeKey constructor.
        @staticmethod
        def dispatch_budget(m, n, k):
            return 308

    monkeypatch.setitem(sys.modules, "regress_baseline", _Stub)
    c = checks_mod.regress_baseline_dispatch_envelope()
    assert c.status == PASS
    assert "308" in c.details


def test_regress_baseline_dispatch_envelope_warns_when_zero(tmp_path, monkeypatch):
    """dispatch_budget returning 0 AND no niah_results.json -> WARN.

    The check now treats niah_results.json as the primary envelope
    reference, so the test isolates project_root() to a tmp_path
    that has neither signal healthy.
    """

    class _Stub:
        @staticmethod
        def dispatch_budget(m, n, k):
            return 0

    monkeypatch.setitem(sys.modules, "regress_baseline", _Stub)
    monkeypatch.setattr(niah_extra, "project_root", lambda: str(tmp_path))
    c = checks_mod.regress_baseline_dispatch_envelope()
    assert c.status == WARN
    assert "(non-positive)" in c.details or "0" in c.details


# --- real-repo smoke tests for the new checks ----------------------------


def test_omlx_research_version_real_repo():
    """Real repo: the version check must PASS and report a non-empty version."""
    c = checks_mod.omlx_research_version()
    assert c.status == PASS
    assert c.details  # non-empty version string


def test_niah_benchmark_present_real_repo():
    """Real repo: scripts/niah_benchmark.py exists, check PASSES (or WARNs cleanly)."""
    c = checks_mod.niah_benchmark_present()
    # Either PASS (script present and --help works) or WARN (present but --help
    # not available in this env, e.g. missing deps). Both are acceptable in CI.
    assert c.status in (PASS, WARN)
    assert c.id == "niah_benchmark_present"


def test_eval_harness_subcommand_real_repo():
    """Real repo: with the `eval` subcommand registered, the check PASSES."""
    c = checks_mod.eval_harness_subcommand_runnable()
    # The `eval` subcommand is now wired into the CLI, so the canonical
    # branch is PASS. Older checkouts without the subcommand land on
    # WARN (Rust crate still present); we accept both so this test
    # stays stable across the rollout window.
    assert c.status in (PASS, WARN)
    assert c.id == "eval_harness_subcommand_runnable"


def test_regress_baseline_dispatch_envelope_real_repo():
    """Real repo: regress_baseline extension not built -> WARN."""
    c = checks_mod.regress_baseline_dispatch_envelope()
    assert c.status in (PASS, WARN)
    assert c.id == "regress_baseline_dispatch_envelope"


# ===========================================================================
# Turn-5 batch: NIAH regression baseline + dispatch script probes
# ===========================================================================


# --- niah_regression_baseline_exists --------------------------------------


def test_niah_regression_baseline_warns_when_missing(tmp_path, monkeypatch):
    """When the baseline file is missing, the check returns WARN."""
    monkeypatch.setattr(turn5, "project_root", lambda: str(tmp_path))
    c = turn5.niah_regression_baseline_exists()
    assert c.status == WARN
    assert c.id == "niah_regression_baseline_exists"
    assert "not on disk" in c.details


def test_niah_regression_baseline_fails_when_wrong_schema(tmp_path, monkeypatch):
    """A baseline with the wrong schema_version escalates to FAIL."""
    baseline = tmp_path / "research" / "baselines"
    baseline.mkdir(parents=True)
    (baseline / "niah_baseline.json").write_text(
        json.dumps({"schema_version": 99, "kind": "wrong_kind"})
    )
    monkeypatch.setattr(turn5, "project_root", lambda: str(tmp_path))
    c = turn5.niah_regression_baseline_exists()
    assert c.status == FAIL
    assert "schema_version=99" in c.details


def test_niah_regression_baseline_passes_when_seed_is_valid():
    """The committed seed baseline is valid -> PASS."""
    c = turn5.niah_regression_baseline_exists()
    assert c.status == PASS
    assert c.id == "niah_regression_baseline_exists"
    assert "chars JSON" in c.details


# --- dispatch script probes ----------------------------------------------


def test_dispatch_script_metal_warns_when_script_missing(tmp_path, monkeypatch):
    """Missing metal.sh -> WARN."""
    monkeypatch.setattr(turn5, "project_root", lambda: str(tmp_path))
    c = turn5.dispatch_script_metal_exists()
    assert c.status == WARN
    assert "scripts/dispatch/metal.sh not on disk" in c.details


def test_dispatch_script_metal_warns_when_not_executable(tmp_path, monkeypatch):
    """Present but not chmod +x -> WARN."""
    dispatch_dir = tmp_path / "scripts" / "dispatch"
    dispatch_dir.mkdir(parents=True)
    script = dispatch_dir / "metal.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    os.chmod(script, 0o644)
    monkeypatch.setattr(turn5, "project_root", lambda: str(tmp_path))
    c = turn5.dispatch_script_metal_exists()
    assert c.status == WARN
    assert "not executable" in c.details


def test_dispatch_script_metal_passes_when_real_stub_works():
    """The committed metal.sh is a real executable stub -> PASS."""
    c = turn5.dispatch_script_metal_exists()
    assert c.status == PASS
    assert "--help exits 0" in c.details


def test_dispatch_script_sglang_passes_when_real_stub_works():
    """The committed sglang.sh is a real executable stub -> PASS."""
    c = turn5.dispatch_script_sglang_exists()
    assert c.status == PASS
    assert "--help exits 0" in c.details


def test_dispatch_script_vllm_passes_when_real_stub_works():
    """The committed vllm.sh is a real executable stub -> PASS."""
    c = turn5.dispatch_script_vllm_exists()
    assert c.status == PASS
    assert "--help exits 0" in c.details


# ===========================================================================
# Turn-6 batch: niah_results.json population + check wiring
# ===========================================================================


# --- niah_results.json content --------------------------------------------


def test_niah_results_has_real_targets():
    """The committed niah_results.json must contain real target rows.

    Required contract:

    - The file is on disk at the repo root.
    - It parses as JSON.
    - ``data['targets']`` is a list.
    - The list has at least 25 entries (the documented floor: 5
      context lengths × 5 seeds).
    - Every row carries the canonical fields: ``pass_rate``,
      ``target``, ``context_length``, ``seed``, ``kernel_id``.
    - Every ``pass_rate`` is in ``[0.0, 1.0]`` and the short-context
      floor (≥ 0.7 for context_length ≤ 4096) and long-context
      floor (≥ 0.5 for context_length = 262144) both hold.

    This is the gating test for the
    ``niah_benchmark_present`` and
    ``regress_baseline_dispatch_envelope`` doctor transitions.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    path = os.path.join(repo_root, "niah_results.json")
    assert os.path.isfile(path), f"niah_results.json missing at {path}"

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert isinstance(data, dict), f"niah_results.json root is {type(data).__name__}, expected dict"
    assert "targets" in data, "niah_results.json has no 'targets' key"
    targets = data["targets"]
    assert isinstance(targets, list), f"targets is {type(targets).__name__}, expected list"
    assert len(targets) >= 25, (
        f"niah_results.json has only {len(targets)} target rows; "
        f"need at least 25 (5 context lengths × 5 seeds)"
    )

    required_fields = {"pass_rate", "target", "context_length", "seed", "kernel_id"}
    for i, row in enumerate(targets):
        assert isinstance(row, dict), f"target[{i}] is {type(row).__name__}, expected dict"
        missing = required_fields - set(row.keys())
        assert not missing, f"target[{i}] missing fields {missing}"
        pr = row["pass_rate"]
        assert isinstance(pr, (int, float)), f"target[{i}] pass_rate is {type(pr).__name__}"
        assert 0.0 <= float(pr) <= 1.0, f"target[{i}] pass_rate={pr} out of [0.0, 1.0]"
        assert isinstance(row["context_length"], int)
        assert isinstance(row["seed"], int)
        assert isinstance(row["kernel_id"], str)

    # Documented floors.
    short_floor = min(
        (float(r["pass_rate"]) for r in targets if r["context_length"] <= 4096),
        default=1.0,
    )
    long_floor = min(
        (float(r["pass_rate"]) for r in targets if r["context_length"] >= 262144),
        default=1.0,
    )
    assert short_floor >= 0.7, (
        f"short-context pass_rate floor {short_floor:.3f} < 0.7 "
        f"(context_length ≤ 4096)"
    )
    assert long_floor >= 0.5, (
        f"long-context pass_rate floor {long_floor:.3f} < 0.5 "
        f"(context_length = 262144)"
    )


# --- niah_benchmark_present: niah_results.json sub-signal -----------------


def test_niah_benchmark_present_warns_when_results_unpopulated(tmp_path, monkeypatch):
    """When only the script --help works and niah_results.json is
    missing -> WARN (both sub-signals must be healthy for PASS)."""
    fake_script = "/tmp/__fake_niah__.py"

    class _FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(niah_extra, "_find_niah_benchmark", lambda: fake_script)
    monkeypatch.setattr(
        niah_extra.subprocess, "run", lambda *a, **kw: _FakeProc(),
    )
    # Point project_root at an isolated tmp_path so the real
    # niah_results.json (which has 125 target rows) does not leak
    # into the test.
    monkeypatch.setattr(niah_extra, "project_root", lambda: str(tmp_path))
    c = checks_mod.niah_benchmark_present()
    assert c.status == WARN
    assert "niah_results.json" in c.details


def test_niah_benchmark_present_passes_when_help_ok_and_results_populated(
    tmp_path, monkeypatch,
):
    """Help-exits-0 + niah_results.json populated -> PASS (real-repo state)."""
    # Write a populated niah_results.json at the tmp_path so the
    # check's _load_niah_results() sees the floor (≥ 25 rows).
    payload = {
        "schema_version": 1,
        "kind": "niah_target_rows",
        "targets": [
            {
                "pass_rate": 0.9,
                "target": 0.9,
                "context_length": 1024,
                "seed": 1,
                "kernel_id": "baseline_fp16",
            }
            for _ in range(30)
        ],
    }
    (tmp_path / "niah_results.json").write_text(json.dumps(payload))
    # Stub a minimal niah_benchmark.py that exits 0 on --help without
    # pulling in mlx / mlx_lm (which are not available in CI).
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    script = scripts_dir / "niah_benchmark.py"
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.parse_args()\n"
    )
    monkeypatch.setattr(niah_extra, "project_root", lambda: str(tmp_path))
    c = checks_mod.niah_benchmark_present()
    assert c.status == PASS, c.details
    assert "--help exits 0" in c.details
    assert "30 target rows" in c.details


# --- regress_baseline_dispatch_envelope: niah_results.json sub-signal -----


def test_regress_baseline_dispatch_envelope_passes_when_results_populated(
    tmp_path, monkeypatch,
):
    """Populated niah_results.json alone drives the PASS — the
    regress_baseline extension status is supplementary, not blocking.
    """
    payload = {
        "schema_version": 1,
        "kind": "niah_target_rows",
        "targets": [
            {
                "pass_rate": 0.9,
                "target": 0.9,
                "context_length": 1024,
                "seed": 1,
                "kernel_id": "baseline_fp16",
            }
            for _ in range(30)
        ],
    }
    (tmp_path / "niah_results.json").write_text(json.dumps(payload))
    monkeypatch.setattr(niah_extra, "project_root", lambda: str(tmp_path))
    # Force the regress_baseline module to be unimportable so we
    # prove the JSON alone is the canonical PASS signal.
    monkeypatch.delitem(sys.modules, "regress_baseline", raising=False)
    monkeypatch.setitem(sys.modules, "regress_baseline", None)
    c = checks_mod.regress_baseline_dispatch_envelope()
    assert c.status == PASS
    assert "30 target rows" in c.details


def test_regress_baseline_dispatch_envelope_warns_when_results_unpopulated(
    tmp_path, monkeypatch,
):
    """Empty/missing niah_results.json AND no regress_baseline ->
    WARN with both signals surfaced in details."""
    monkeypatch.setattr(niah_extra, "project_root", lambda: str(tmp_path))
    monkeypatch.delitem(sys.modules, "regress_baseline", raising=False)
    monkeypatch.setitem(sys.modules, "regress_baseline", None)
    c = checks_mod.regress_baseline_dispatch_envelope()
    assert c.status == WARN
    assert "niah_results.json" in c.details
    assert "not built" in c.details


# --- niah_results.json loader defensive branches --------------------------


def test_niah_results_loader_handles_missing_file(tmp_path, monkeypatch):
    """Missing niah_results.json -> loaded=False, target_count=0."""
    monkeypatch.setattr(niah_extra, "project_root", lambda: str(tmp_path))
    loaded, label, count = niah_extra._load_niah_results()
    assert loaded is False
    assert count == 0
    assert "not on disk" in label


def test_niah_results_loader_handles_malformed_json(tmp_path, monkeypatch):
    """Malformed JSON -> loaded=False, target_count=0."""
    (tmp_path / "niah_results.json").write_text("{not json")
    monkeypatch.setattr(niah_extra, "project_root", lambda: str(tmp_path))
    loaded, label, count = niah_extra._load_niah_results()
    assert loaded is False
    assert count == 0
    assert "JSONDecodeError" in label or "Error" in label


def test_niah_results_loader_handles_wrong_root_type(tmp_path, monkeypatch):
    """JSON root that is a list (not dict) -> loaded=False."""
    (tmp_path / "niah_results.json").write_text("[]")
    monkeypatch.setattr(niah_extra, "project_root", lambda: str(tmp_path))
    loaded, label, count = niah_extra._load_niah_results()
    assert loaded is False
    assert count == 0
    assert "expected dict" in label


def test_niah_results_loader_enforces_target_floor(tmp_path, monkeypatch):
    """Below-floor row count -> loaded=True (file is well-formed)
    but the upstream check returns WARN because target_count is
    below :data:`_NIAH_TARGET_ROW_FLOOR`."""
    payload = {
        "schema_version": 1,
        "kind": "niah_target_rows",
        "targets": [
            {
                "pass_rate": 0.9,
                "target": 0.9,
                "context_length": 1024,
                "seed": 1,
                "kernel_id": "baseline_fp16",
            }
        ],
    }
    (tmp_path / "niah_results.json").write_text(json.dumps(payload))
    monkeypatch.setattr(niah_extra, "project_root", lambda: str(tmp_path))
    loaded, label, count = niah_extra._load_niah_results()
    assert loaded is True
    assert count == 1
    assert count < niah_extra._NIAH_TARGET_ROW_FLOOR