"""Tests for the carved-out half of the doctor internal checks.

The original :mod:`omlx_research.cli._doctor_internal_checks` module
shipped four turn-10 checks in a single 576-line file. After turn-12
discovered this was over the 500-line cap, two of the four checks
(``metal_runtime_lib_test_count_at_least_25`` and
``python_cli_subcommand_count_at_least_6``) were carved out into the
sibling module :mod:`omlx_research.cli._doctor_internal_checks_split`.
This file exercises those two carved-out checks.

The other two turn-10 checks (``coverage_tag_count_at_least_25`` and
``eval_harness_suite_count_at_least_4``) stayed in the original module
and are covered by ``test_doctor_internal_checks.py``.

These checks inspect on-disk source files in the repo (never external
services) and verify structural invariants:

1. ``metal_runtime_lib_test_count_at_least_25`` —
   ``perf-core/metal-runtime/src/`` carries ≥25 distinct ``#[test]``
   attributes across all its source files (fails <15, warns <25).
2. ``python_cli_subcommand_count_at_least_6`` —
   ``python/omlx_research/cli/__init__.py`` registers ≥6 distinct
   ``cmd_*`` callable subcommands (fails <4, warns <6).

The tests exercise both the threshold-ladder behavior (via
``_patch_count`` / ``_patch_missing`` helpers that monkeypatch the
private counters without touching the on-disk files) and the
graceful-degradation paths (file missing → WARN, never FAIL).
"""

from __future__ import annotations

from omlx_research.cli import _doctor_internal_checks_split as split_mod
from omlx_research.cli._doctor_shared import FAIL, PASS, WARN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_count(
    monkeypatch,
    *,
    metal_runtime: int | None = None,
    cli: int | None = None,
):
    """Force the private counters to return the supplied integers.

    Mirrors the ``_patch_count`` helper in ``test_doctor_internal_checks.py``:
    we monkeypatch the internal counters so the threshold-ladder tests
    run in isolation without touching the on-disk source files. All
    args are optional; only the ones supplied are patched.

    Only the two checks that live in
    :mod:`omlx_research.cli._doctor_internal_checks_split` are patchable
    here. The other two turn-10 checks live in the original
    :mod:`omlx_research.cli._doctor_internal_checks` and are exercised
    via the sibling test file ``test_doctor_internal_checks.py``.
    """
    if metal_runtime is not None:
        monkeypatch.setattr(
            split_mod, "_count_metal_runtime_tests",
            lambda _path: (True, f"found {metal_runtime} distinct #[test] reference(s) under src"),
        )
    if cli is not None:
        monkeypatch.setattr(
            split_mod, "_count_cli_subcommands",
            lambda _path: (True, f"found {cli} distinct cmd_* subcommand(s)"),
        )


def _patch_missing(monkeypatch, target: str) -> None:
    """Force the named counter to behave as if the file/dir is missing.

    ``target`` is ``"metal_runtime"`` or ``"cli"``. Mirrors the real
    on-disk ``not on disk`` failure mode for those two checks. The
    ``"coverage"`` and ``"eval"`` targets live with the checks that
    stayed in ``_doctor_internal_checks.py``; their tests live in
    ``test_doctor_internal_checks.py``.
    """
    if target == "metal_runtime":
        monkeypatch.setattr(
            split_mod, "_count_metal_runtime_tests",
            lambda _path: (False, "perf-core/metal-runtime/src not on disk"),
        )
    elif target == "cli":
        monkeypatch.setattr(
            split_mod, "_count_cli_subcommands",
            lambda _path: (False, "python/omlx_research/cli/__init__.py not on disk"),
        )
    else:
        raise AssertionError(f"unknown target {target!r}")


# ---------------------------------------------------------------------------
# metal_runtime_lib_test_count_at_least_25 — threshold ladder
# ---------------------------------------------------------------------------


def test_metal_runtime_pass_at_threshold(monkeypatch):
    """count == 25 → PASS (boundary: >= 25)."""
    _patch_count(monkeypatch, metal_runtime=25)
    c = split_mod.metal_runtime_lib_test_count_at_least_25()
    assert c.status == PASS
    assert c.id == "metal_runtime_lib_test_count_at_least_25"
    assert "25" in c.details


def test_metal_runtime_pass_above_threshold(monkeypatch):
    """count > 25 → PASS (the real src/ currently reports higher)."""
    _patch_count(monkeypatch, metal_runtime=40)
    c = split_mod.metal_runtime_lib_test_count_at_least_25()
    assert c.status == PASS
    assert "40" in c.details


def test_metal_runtime_warn_in_band(monkeypatch):
    """count == 20 (between 15 and 24) → WARN."""
    _patch_count(monkeypatch, metal_runtime=20)
    c = split_mod.metal_runtime_lib_test_count_at_least_25()
    assert c.status == WARN
    assert "20" in c.details


def test_metal_runtime_warn_at_upper_boundary(monkeypatch):
    """count == 24 → WARN (boundary: < 25)."""
    _patch_count(monkeypatch, metal_runtime=24)
    c = split_mod.metal_runtime_lib_test_count_at_least_25()
    assert c.status == WARN


def test_metal_runtime_warn_at_lower_boundary(monkeypatch):
    """count == 15 → WARN (boundary: >= 15 and < 25)."""
    _patch_count(monkeypatch, metal_runtime=15)
    c = split_mod.metal_runtime_lib_test_count_at_least_25()
    assert c.status == WARN


def test_metal_runtime_fail_below_floor(monkeypatch):
    """count == 10 (< 15) → FAIL."""
    _patch_count(monkeypatch, metal_runtime=10)
    c = split_mod.metal_runtime_lib_test_count_at_least_25()
    assert c.status == FAIL
    assert "10" in c.details


def test_metal_runtime_fail_at_zero(monkeypatch):
    """count == 0 → FAIL (the directory would be empty / unreadable)."""
    _patch_count(monkeypatch, metal_runtime=0)
    c = split_mod.metal_runtime_lib_test_count_at_least_25()
    assert c.status == FAIL


# ---------------------------------------------------------------------------
# metal_runtime_lib_test_count_at_least_25 — graceful degradation
# ---------------------------------------------------------------------------


def test_metal_runtime_warns_when_dir_missing(monkeypatch):
    """Missing metal-runtime src/ directory → WARN, never FAIL."""
    _patch_missing(monkeypatch, "metal_runtime")
    c = split_mod.metal_runtime_lib_test_count_at_least_25()
    assert c.status == WARN
    assert c.id == "metal_runtime_lib_test_count_at_least_25"
    assert "not on disk" in c.details
    assert "never FAIL" in c.details


def test_metal_runtime_warns_on_oserror(monkeypatch):
    """OS-level read failure → WARN (defensive)."""
    monkeypatch.setattr(
        split_mod, "_count_metal_runtime_tests",
        lambda _path: (False, "PermissionError: [Errno 13] Permission denied"),
    )
    c = split_mod.metal_runtime_lib_test_count_at_least_25()
    assert c.status == WARN
    assert "PermissionError" in c.details


# ---------------------------------------------------------------------------
# python_cli_subcommand_count_at_least_6 — threshold ladder
# ---------------------------------------------------------------------------


def test_cli_subcmd_pass_at_threshold(monkeypatch):
    """count == 6 → PASS (boundary: >= 6)."""
    _patch_count(monkeypatch, cli=6)
    c = split_mod.python_cli_subcommand_count_at_least_6()
    assert c.status == PASS
    assert c.id == "python_cli_subcommand_count_at_least_6"
    assert "6" in c.details


def test_cli_subcmd_pass_above_threshold(monkeypatch):
    """count > 6 → PASS (room to grow)."""
    _patch_count(monkeypatch, cli=8)
    c = split_mod.python_cli_subcommand_count_at_least_6()
    assert c.status == PASS
    assert "8" in c.details


def test_cli_subcmd_warn_in_band(monkeypatch):
    """count == 5 (between 4 and 5) → WARN."""
    _patch_count(monkeypatch, cli=5)
    c = split_mod.python_cli_subcommand_count_at_least_6()
    assert c.status == WARN
    assert "5" in c.details


def test_cli_subcmd_warn_at_lower_boundary(monkeypatch):
    """count == 4 → WARN (boundary: >= 4 and < 6)."""
    _patch_count(monkeypatch, cli=4)
    c = split_mod.python_cli_subcommand_count_at_least_6()
    assert c.status == WARN


def test_cli_subcmd_fail_below_floor(monkeypatch):
    """count == 2 (< 4) → FAIL."""
    _patch_count(monkeypatch, cli=2)
    c = split_mod.python_cli_subcommand_count_at_least_6()
    assert c.status == FAIL
    assert "2" in c.details


def test_cli_subcmd_fail_at_zero(monkeypatch):
    """count == 0 → FAIL (no subcommands registered at all)."""
    _patch_count(monkeypatch, cli=0)
    c = split_mod.python_cli_subcommand_count_at_least_6()
    assert c.status == FAIL


# ---------------------------------------------------------------------------
# python_cli_subcommand_count_at_least_6 — graceful degradation
# ---------------------------------------------------------------------------


def test_cli_subcmd_warns_when_file_missing(monkeypatch):
    """Missing __init__.py → WARN, never FAIL."""
    _patch_missing(monkeypatch, "cli")
    c = split_mod.python_cli_subcommand_count_at_least_6()
    assert c.status == WARN
    assert c.id == "python_cli_subcommand_count_at_least_6"
    assert "not on disk" in c.details
    assert "never FAIL" in c.details


def test_cli_subcmd_warns_on_oserror(monkeypatch):
    """OS-level read failure → WARN (defensive)."""
    monkeypatch.setattr(
        split_mod, "_count_cli_subcommands",
        lambda _path: (False, "OSError: [Errno 2] No such file or directory"),
    )
    c = split_mod.python_cli_subcommand_count_at_least_6()
    assert c.status == WARN
    assert "OSError" in c.details


# ---------------------------------------------------------------------------
# Live (non-mocked) sanity check — confirms the checks are wired into the
# real registry AND that the live repo state passes cleanly.
# ---------------------------------------------------------------------------


def test_metal_runtime_passes_on_real_repo():
    """The real metal-runtime src/ must clear the ≥25 floor.

    Real-on-disk run; no mocking. If this fails the threshold ladder
    was calibrated against a stale snapshot or the carve-out broke
    the import wiring.
    """
    c = split_mod.metal_runtime_lib_test_count_at_least_25()
    assert c.status == PASS, (
        f"real metal-runtime did not clear the floor: {c.details}"
    )
    assert c.id == "metal_runtime_lib_test_count_at_least_25"


def test_cli_subcmd_passes_on_real_repo():
    """The real CLI __init__.py must register ≥6 distinct cmd_*."""
    c = split_mod.python_cli_subcommand_count_at_least_6()
    assert c.status == PASS, (
        f"real CLI did not expose 6 subcommands: {c.details}"
    )
    assert c.id == "python_cli_subcommand_count_at_least_6"


def test_both_split_checks_are_registered_in_doctor_checks():
    """Both carved-out check callables must be present in :data:`doctor.CHECKS`.

    After the turn-10 → split batch move, both checks live in
    :mod:`omlx_research.cli._doctor_internal_checks_split` and are
    re-exported via :mod:`omlx_research.cli._doctor_checks`.
    """
    from omlx_research.cli.doctor import CHECKS

    ids = {getattr(fn, "__name__", "") for fn in CHECKS}
    assert "metal_runtime_lib_test_count_at_least_25" in ids
    assert "python_cli_subcommand_count_at_least_6" in ids
