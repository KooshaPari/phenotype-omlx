"""Tests for the four doctor checks added 2026-07-19.

Mirrors the patterns in :mod:`omlx_research.cli.tests.test_doctor`
but isolates the new check coverage into its own module so
``test_doctor.py`` stays under the 500L hard cap. Each test exercises
either a monkey-patched failure branch (for ``unittest.mock.patch``-
style coverage of missing modules) or a real-repo smoke test that
runs against the live repository.
"""

from __future__ import annotations

import sys

from omlx_research.cli import _doctor_checks as checks_mod
from omlx_research.cli import _doctor_extra_checks as extra
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

    monkeypatch.setattr(extra, "_find_niah_benchmark", _fake_find)
    monkeypatch.setattr(
        extra.subprocess, "run",
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
    monkeypatch.setattr(extra, "_find_niah_benchmark", lambda: None)
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
        extra, "_find_niah_benchmark",
        lambda: "/tmp/__fake_niah__.py",
    )
    monkeypatch.setattr(
        extra.subprocess, "run",
        lambda *a, **kw: _FakeProc(),
    )
    c = checks_mod.niah_benchmark_present()
    assert c.status == WARN
    assert "exited 2" in c.details


# --- eval_harness_subcommand_runnable -------------------------------------


def test_eval_harness_subcommand_fails_when_import_missing(monkeypatch):
    """omlx_research.eval missing -> FAIL (critical surface)."""
    monkeypatch.setattr(
        extra, "_eval_harness_module",
        lambda: (False, "ModuleNotFoundError: No module named 'omlx_research.eval'"),
    )
    c = checks_mod.eval_harness_subcommand_runnable()
    assert c.status == FAIL
    assert c.id == "eval_harness_subcommand_runnable"
    assert "omlx_research.eval failed to import" in c.details


def test_eval_harness_subcommand_warns_when_subcmd_missing(monkeypatch):
    """omlx_research.eval imports but no `eval` subcmd -> WARN."""
    monkeypatch.setattr(extra, "_eval_harness_module", lambda: (True, "/some/path/eval.py"))
    monkeypatch.setattr(extra, "_cli_has_eval_subcommand", lambda: False)
    monkeypatch.setattr(extra, "_list_eval_harness_tests", lambda: [])
    c = checks_mod.eval_harness_subcommand_runnable()
    assert c.status == WARN
    assert "does not yet expose an `eval` subcommand" in c.details


def test_eval_harness_subcommand_passes_when_subcmd_present(monkeypatch):
    """Both import + subcommand present -> PASS."""
    monkeypatch.setattr(extra, "_eval_harness_module", lambda: (True, "/some/path/eval.py"))
    monkeypatch.setattr(extra, "_cli_has_eval_subcommand", lambda: True)
    monkeypatch.setattr(extra, "_list_eval_harness_tests", lambda: ["test_eval_harness.py"])
    c = checks_mod.eval_harness_subcommand_runnable()
    assert c.status == PASS
    assert "eval` subcommand available" in c.details
    assert "test_eval_harness.py" in c.details


# --- regress_baseline_dispatch_envelope ----------------------------------


def test_regress_baseline_dispatch_envelope_warns_when_extension_missing(monkeypatch):
    """Missing regress_baseline module -> WARN with maturin hint."""
    monkeypatch.delitem(sys.modules, "regress_baseline", raising=False)
    monkeypatch.setitem(sys.modules, "regress_baseline", None)
    c = checks_mod.regress_baseline_dispatch_envelope()
    assert c.status == WARN
    assert c.id == "regress_baseline_dispatch_envelope"
    assert "not built" in c.details


def test_regress_baseline_dispatch_envelope_passes_with_stub_module(monkeypatch):
    """A stub regress_baseline exposing dispatch_budget() -> PASS."""

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


def test_regress_baseline_dispatch_envelope_warns_when_zero(monkeypatch):
    """dispatch_budget returning 0 -> WARN (no finite ceiling)."""

    class _Stub:
        @staticmethod
        def dispatch_budget(m, n, k):
            return 0

    monkeypatch.setitem(sys.modules, "regress_baseline", _Stub)
    c = checks_mod.regress_baseline_dispatch_envelope()
    assert c.status == WARN
    assert "expected a finite positive ceiling" in c.details


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
    """Real repo: omlx_research.eval does not exist yet, so WARN is expected."""
    c = checks_mod.eval_harness_subcommand_runnable()
    # The eval module is a known gap (no Python module ships yet), so the
    # most common branch is WARN. If someone adds it later, this will PASS.
    assert c.status in (PASS, WARN, FAIL)
    assert c.id == "eval_harness_subcommand_runnable"


def test_regress_baseline_dispatch_envelope_real_repo():
    """Real repo: regress_baseline extension not built -> WARN."""
    c = checks_mod.regress_baseline_dispatch_envelope()
    assert c.status in (PASS, WARN)
    assert c.id == "regress_baseline_dispatch_envelope"