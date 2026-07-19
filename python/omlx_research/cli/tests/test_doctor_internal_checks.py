"""Tests for the doctor internal structural-invariant checks (turn-10).

These checks inspect on-disk source files in the repo (never external
services) and verify two structural invariants:

1. ``coverage_tag_count_at_least_25`` — ``coverage_matrix.rs`` carries
   ≥25 distinct tag-style declarations (fails <15, warns <25).
2. ``eval_harness_suite_count_at_least_4`` — ``eval-harness/src/lib.rs``
   exposes ≥4 distinct ``Suite::Variant`` references (fails <2,
   warns <4).

The tests exercise both the threshold-ladder behavior (via
``_patch_count`` / ``_patch_path`` helpers that monkeypatch the
private counters without touching the on-disk files) and the
graceful-degradation paths (file missing → WARN, never FAIL).
"""

from __future__ import annotations

import os

from omlx_research.cli import _doctor_internal_checks as internal_mod
from omlx_research.cli._doctor_shared import FAIL, PASS, WARN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_count(monkeypatch, *, coverage: int | None = None, eval_suites: int | None = None):
    """Force the private counters to return the supplied integers.

    Mirrors the ``_patch_threshold`` helper in ``test_doctor_meta.py``:
    we monkeypatch the internal counters so the threshold-ladder tests
    run in isolation without touching the on-disk source files. Both
    args are optional; only the ones supplied are patched.
    """
    if coverage is not None:
        monkeypatch.setattr(
            internal_mod, "_count_coverage_tags",
            lambda _path: (True, f"found {coverage} distinct tag(s) in coverage_matrix.rs"),
        )
    if eval_suites is not None:
        monkeypatch.setattr(
            internal_mod, "_count_eval_suites",
            lambda _path: (True, f"found {eval_suites} distinct Suite variant(s)"),
        )


def _patch_missing(monkeypatch, target: str) -> None:
    """Force the named counter to behave as if the file is missing.

    ``target`` is ``"coverage"`` or ``"eval"``. Mirrors the real
    on-disk ``not on disk`` failure mode.
    """
    if target == "coverage":
        monkeypatch.setattr(
            internal_mod, "_count_coverage_tags",
            lambda _path: (False, "perf-core/kernel-registry/tests/sota_operators/coverage_matrix.rs not on disk"),
        )
    elif target == "eval":
        monkeypatch.setattr(
            internal_mod, "_count_eval_suites",
            lambda _path: (False, "perf-core/eval-harness/src/lib.rs not on disk"),
        )
    else:
        raise AssertionError(f"unknown target {target!r}")


# ---------------------------------------------------------------------------
# coverage_tag_count_at_least_25 — threshold ladder
# ---------------------------------------------------------------------------


def test_coverage_tag_pass_at_threshold(monkeypatch):
    """count == 25 → PASS (boundary: >= 25)."""
    _patch_count(monkeypatch, coverage=25)
    c = internal_mod.coverage_tag_count_at_least_25()
    assert c.status == PASS
    assert c.id == "coverage_tag_count_at_least_25"
    assert "25" in c.details


def test_coverage_tag_pass_above_threshold(monkeypatch):
    """count > 25 → PASS (the real file currently reports 66)."""
    _patch_count(monkeypatch, coverage=66)
    c = internal_mod.coverage_tag_count_at_least_25()
    assert c.status == PASS
    assert "66" in c.details


def test_coverage_tag_warn_in_band(monkeypatch):
    """count == 20 (between 15 and 24) → WARN."""
    _patch_count(monkeypatch, coverage=20)
    c = internal_mod.coverage_tag_count_at_least_25()
    assert c.status == WARN
    assert "20" in c.details


def test_coverage_tag_warn_at_upper_boundary(monkeypatch):
    """count == 24 → WARN (boundary: < 25)."""
    _patch_count(monkeypatch, coverage=24)
    c = internal_mod.coverage_tag_count_at_least_25()
    assert c.status == WARN


def test_coverage_tag_warn_at_lower_boundary(monkeypatch):
    """count == 15 → WARN (boundary: >= 15 and < 25)."""
    _patch_count(monkeypatch, coverage=15)
    c = internal_mod.coverage_tag_count_at_least_25()
    assert c.status == WARN


def test_coverage_tag_fail_below_floor(monkeypatch):
    """count == 10 (< 15) → FAIL."""
    _patch_count(monkeypatch, coverage=10)
    c = internal_mod.coverage_tag_count_at_least_25()
    assert c.status == FAIL
    assert "10" in c.details


def test_coverage_tag_fail_at_zero(monkeypatch):
    """count == 0 → FAIL (the file would be empty / unreadable)."""
    _patch_count(monkeypatch, coverage=0)
    c = internal_mod.coverage_tag_count_at_least_25()
    assert c.status == FAIL


# ---------------------------------------------------------------------------
# coverage_tag_count_at_least_25 — graceful degradation
# ---------------------------------------------------------------------------


def test_coverage_tag_warns_when_file_missing(monkeypatch):
    """Missing coverage matrix file → WARN, never FAIL."""
    _patch_missing(monkeypatch, "coverage")
    c = internal_mod.coverage_tag_count_at_least_25()
    assert c.status == WARN
    assert c.id == "coverage_tag_count_at_least_25"
    assert "not on disk" in c.details
    assert "never FAIL" in c.details


def test_coverage_tag_warns_on_oserror(monkeypatch):
    """OS-level read failure → WARN (defensive)."""
    monkeypatch.setattr(
        internal_mod, "_count_coverage_tags",
        lambda _path: (False, "PermissionError: [Errno 13] Permission denied"),
    )
    c = internal_mod.coverage_tag_count_at_least_25()
    assert c.status == WARN
    assert "PermissionError" in c.details


# ---------------------------------------------------------------------------
# eval_harness_suite_count_at_least_4 — threshold ladder
# ---------------------------------------------------------------------------


def test_eval_suite_pass_at_threshold(monkeypatch):
    """count == 4 → PASS (boundary: >= 4)."""
    _patch_count(monkeypatch, eval_suites=4)
    c = internal_mod.eval_harness_suite_count_at_least_4()
    assert c.status == PASS
    assert c.id == "eval_harness_suite_count_at_least_4"
    assert "4" in c.details


def test_eval_suite_pass_above_threshold(monkeypatch):
    """count > 4 → PASS (room to grow)."""
    _patch_count(monkeypatch, eval_suites=6)
    c = internal_mod.eval_harness_suite_count_at_least_4()
    assert c.status == PASS
    assert "6" in c.details


def test_eval_suite_warn_in_band(monkeypatch):
    """count == 3 (between 2 and 3) → WARN."""
    _patch_count(monkeypatch, eval_suites=3)
    c = internal_mod.eval_harness_suite_count_at_least_4()
    assert c.status == WARN
    assert "3" in c.details


def test_eval_suite_warn_at_lower_boundary(monkeypatch):
    """count == 2 → WARN (boundary: >= 2 and < 4)."""
    _patch_count(monkeypatch, eval_suites=2)
    c = internal_mod.eval_harness_suite_count_at_least_4()
    assert c.status == WARN


def test_eval_suite_fail_below_floor(monkeypatch):
    """count == 1 (< 2) → FAIL."""
    _patch_count(monkeypatch, eval_suites=1)
    c = internal_mod.eval_harness_suite_count_at_least_4()
    assert c.status == FAIL
    assert "1" in c.details


def test_eval_suite_fail_at_zero(monkeypatch):
    """count == 0 → FAIL (no Suite enum at all)."""
    _patch_count(monkeypatch, eval_suites=0)
    c = internal_mod.eval_harness_suite_count_at_least_4()
    assert c.status == FAIL


# ---------------------------------------------------------------------------
# eval_harness_suite_count_at_least_4 — graceful degradation
# ---------------------------------------------------------------------------


def test_eval_suite_warns_when_file_missing(monkeypatch):
    """Missing eval-harness lib.rs → WARN, never FAIL."""
    _patch_missing(monkeypatch, "eval")
    c = internal_mod.eval_harness_suite_count_at_least_4()
    assert c.status == WARN
    assert c.id == "eval_harness_suite_count_at_least_4"
    assert "not on disk" in c.details
    assert "never FAIL" in c.details


def test_eval_suite_warns_on_oserror(monkeypatch):
    """OS-level read failure → WARN (defensive)."""
    monkeypatch.setattr(
        internal_mod, "_count_eval_suites",
        lambda _path: (False, "OSError: [Errno 2] No such file or directory"),
    )
    c = internal_mod.eval_harness_suite_count_at_least_4()
    assert c.status == WARN
    assert "OSError" in c.details


# ---------------------------------------------------------------------------
# Live (non-mocked) sanity check — confirms the checks are wired into the
# real registry AND that the live repo state passes cleanly.
# ---------------------------------------------------------------------------


def test_coverage_tag_passes_on_real_repo():
    """The real coverage_matrix.rs must clear the ≥25 floor.

    Real-on-disk run; no mocking. If this fails the threshold ladder
    was calibrated against a stale snapshot.
    """
    c = internal_mod.coverage_tag_count_at_least_25()
    assert c.status == PASS, (
        f"real coverage matrix did not clear the floor: {c.details}"
    )
    assert c.id == "coverage_tag_count_at_least_25"


def test_eval_suite_passes_on_real_repo():
    """The real eval-harness lib.rs must expose ≥4 distinct Suite variants."""
    c = internal_mod.eval_harness_suite_count_at_least_4()
    assert c.status == PASS, (
        f"real eval-harness did not expose 4 suites: {c.details}"
    )
    assert c.id == "eval_harness_suite_count_at_least_4"


def test_both_internal_checks_are_registered_in_doctor_checks():
    """Both check callables must be present in :data:`doctor.CHECKS`."""
    from omlx_research.cli.doctor import CHECKS

    ids = {getattr(fn, "__name__", "") for fn in CHECKS}
    assert "coverage_tag_count_at_least_25" in ids
    assert "eval_harness_suite_count_at_least_4" in ids