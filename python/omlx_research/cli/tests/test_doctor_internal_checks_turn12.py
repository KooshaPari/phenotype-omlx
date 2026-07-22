"""Tests for the turn-12 doctor internal structural-invariant checks.

These exercise the two checks added on top of the turn-10 batch:

1. ``cargo_workspace_crate_count_at_least_15`` — declares >= 15
   ``[workspace].members`` in ``perf-core/Cargo.toml`` (fails < 10,
   warns < 15).
2. ``ddm_continuous_schedule_variants_at_least_4`` — exposes >= 4
   distinct ``ContinuousScheduleKind`` variants in the discrete
   diffusion oracle (fails < 2, warns < 4).

Both checks are entirely INTERNAL — they inspect on-disk source
files and degrade to WARN — never FAIL — when those files are
missing.

The tests in this module cover:

- the threshold-ladder behavior for both checks (pass/warn/fail
  at the documented boundaries), via monkeypatch on the private
  counters;
- graceful degradation (file missing → WARN, OSError → WARN);
- the real-repo pass (the live crate count is 20; the live
  ContinuousScheduleKind variant count is 4);
- the drift-detector lockstep rule: the live ``doctor --json``
  output contains both new check IDs AND the meta-check passes;
- the threshold-bump doc-comment in ``doctor_config.toml``
  explaining the lockstep rule.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from omlx_research.cli import _doctor_internal_checks_turn12 as turn12_mod
from omlx_research.cli._doctor_shared import FAIL, PASS, WARN
from omlx_research.cli._doctor_shared import project_root as _doctor_project_root


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_count(
    monkeypatch,
    *,
    workspace: int | None = None,
    ddm_variants: int | None = None,
):
    """Force the private counters to return the supplied integers.

    Mirrors ``_patch_count`` in ``test_doctor_internal_checks.py``:
    we monkeypatch the internal counters so the threshold-ladder
    tests run in isolation without touching the on-disk source
    files. Only the counters supplied are patched.
    """
    if workspace is not None:
        monkeypatch.setattr(
            turn12_mod, "_count_workspace_members",
            lambda _path: (
                True,
                f"found {workspace} declared workspace member(s) "
                f"in perf-core/Cargo.toml",
            ),
        )
    if ddm_variants is not None:
        monkeypatch.setattr(
            turn12_mod, "_count_ddm_schedule_variants",
            lambda _path: (
                True,
                f"found {ddm_variants} distinct ContinuousScheduleKind variant(s)",
            ),
        )


def _patch_missing(monkeypatch, target: str) -> None:
    """Force the named counter to behave as if the file is missing.

    ``target`` is ``"workspace"`` or ``"ddm"``. Mirrors the real
    on-disk ``not on disk`` failure mode.
    """
    if target == "workspace":
        monkeypatch.setattr(
            turn12_mod, "_count_workspace_members",
            lambda _path: (False, "perf-core/Cargo.toml not on disk"),
        )
    elif target == "ddm":
        monkeypatch.setattr(
            turn12_mod, "_count_ddm_schedule_variants",
            lambda _path: (
                False,
                "perf-core/kernel-registry/tests/sota_operators/"
                "discrete_diffusion_oracle.rs not on disk",
            ),
        )
    else:
        raise AssertionError(f"unknown target {target!r}")


# ---------------------------------------------------------------------------
# cargo_workspace_crate_count_at_least_15 — threshold ladder
# ---------------------------------------------------------------------------


def test_workspace_crate_pass_at_threshold(monkeypatch):
    """count == 15 → PASS (boundary: >= 15)."""
    _patch_count(monkeypatch, workspace=15)
    c = turn12_mod.cargo_workspace_crate_count_at_least_15()
    assert c.status == PASS
    assert c.id == "cargo_workspace_crate_count_at_least_15"
    assert "15" in c.details


def test_workspace_crate_pass_above_threshold(monkeypatch):
    """count > 15 → PASS (the real workspace currently reports 20)."""
    _patch_count(monkeypatch, workspace=20)
    c = turn12_mod.cargo_workspace_crate_count_at_least_15()
    assert c.status == PASS
    assert "20" in c.details


def test_workspace_crate_warn_in_band(monkeypatch):
    """count == 12 (between 10 and 14) → WARN."""
    _patch_count(monkeypatch, workspace=12)
    c = turn12_mod.cargo_workspace_crate_count_at_least_15()
    assert c.status == WARN
    assert "12" in c.details


def test_workspace_crate_warn_at_upper_boundary(monkeypatch):
    """count == 14 → WARN (boundary: < 15)."""
    _patch_count(monkeypatch, workspace=14)
    c = turn12_mod.cargo_workspace_crate_count_at_least_15()
    assert c.status == WARN


def test_workspace_crate_warn_at_lower_boundary(monkeypatch):
    """count == 10 → WARN (boundary: >= 10 and < 15)."""
    _patch_count(monkeypatch, workspace=10)
    c = turn12_mod.cargo_workspace_crate_count_at_least_15()
    assert c.status == WARN


def test_workspace_crate_fail_below_floor(monkeypatch):
    """count == 8 (< 10) → FAIL."""
    _patch_count(monkeypatch, workspace=8)
    c = turn12_mod.cargo_workspace_crate_count_at_least_15()
    assert c.status == FAIL
    assert "8" in c.details


def test_workspace_crate_fail_at_zero(monkeypatch):
    """count == 0 → FAIL (empty workspace — the eviction accident case)."""
    _patch_count(monkeypatch, workspace=0)
    c = turn12_mod.cargo_workspace_crate_count_at_least_15()
    assert c.status == FAIL


def test_workspace_crate_warns_when_file_missing(monkeypatch):
    """Missing Cargo.toml → WARN, never FAIL."""
    _patch_missing(monkeypatch, "workspace")
    c = turn12_mod.cargo_workspace_crate_count_at_least_15()
    assert c.status == WARN
    assert c.id == "cargo_workspace_crate_count_at_least_15"
    assert "not on disk" in c.details
    assert "never FAIL" in c.details


def test_workspace_crate_warns_on_oserror(monkeypatch):
    """OS-level read failure → WARN (defensive)."""
    monkeypatch.setattr(
        turn12_mod, "_count_workspace_members",
        lambda _path: (False, "PermissionError: [Errno 13] Permission denied"),
    )
    c = turn12_mod.cargo_workspace_crate_count_at_least_15()
    assert c.status == WARN
    assert "PermissionError" in c.details


def test_workspace_crate_passes_on_real_repo():
    """The real Cargo workspace must declare >= 15 members.

    Real-on-disk run; no mocking. The live polyglot workspace has
    20 crates at turn-11 close, well above the 15 threshold. If
    this fails the workspace has shrunk below the floor.
    """
    c = turn12_mod.cargo_workspace_crate_count_at_least_15()
    assert c.status == PASS, (
        f"real workspace did not clear the floor: {c.details}"
    )
    assert c.id == "cargo_workspace_crate_count_at_least_15"


# ---------------------------------------------------------------------------
# ddm_continuous_schedule_variants_at_least_4 — threshold ladder
# ---------------------------------------------------------------------------


def test_ddm_variants_pass_at_threshold(monkeypatch):
    """count == 4 → PASS (boundary: >= 4)."""
    _patch_count(monkeypatch, ddm_variants=4)
    c = turn12_mod.ddm_continuous_schedule_variants_at_least_4()
    assert c.status == PASS
    assert c.id == "ddm_continuous_schedule_variants_at_least_4"
    assert "4" in c.details


def test_ddm_variants_pass_above_threshold(monkeypatch):
    """count > 4 → PASS (room to grow)."""
    _patch_count(monkeypatch, ddm_variants=5)
    c = turn12_mod.ddm_continuous_schedule_variants_at_least_4()
    assert c.status == PASS
    assert "5" in c.details


def test_ddm_variants_warn_in_band(monkeypatch):
    """count == 3 (between 2 and 3) → WARN."""
    _patch_count(monkeypatch, ddm_variants=3)
    c = turn12_mod.ddm_continuous_schedule_variants_at_least_4()
    assert c.status == WARN
    assert "3" in c.details


def test_ddm_variants_warn_at_lower_boundary(monkeypatch):
    """count == 2 → WARN (boundary: >= 2 and < 4)."""
    _patch_count(monkeypatch, ddm_variants=2)
    c = turn12_mod.ddm_continuous_schedule_variants_at_least_4()
    assert c.status == WARN


def test_ddm_variants_fail_below_floor(monkeypatch):
    """count == 1 (< 2) → FAIL."""
    _patch_count(monkeypatch, ddm_variants=1)
    c = turn12_mod.ddm_continuous_schedule_variants_at_least_4()
    assert c.status == FAIL
    assert "1" in c.details


def test_ddm_variants_fail_at_zero(monkeypatch):
    """count == 0 → FAIL (the schedule enum was completely removed)."""
    _patch_count(monkeypatch, ddm_variants=0)
    c = turn12_mod.ddm_continuous_schedule_variants_at_least_4()
    assert c.status == FAIL


def test_ddm_variants_warns_when_file_missing(monkeypatch):
    """Missing oracle file → WARN, never FAIL."""
    _patch_missing(monkeypatch, "ddm")
    c = turn12_mod.ddm_continuous_schedule_variants_at_least_4()
    assert c.status == WARN
    assert c.id == "ddm_continuous_schedule_variants_at_least_4"
    assert "not on disk" in c.details
    assert "never FAIL" in c.details


def test_ddm_variants_warns_on_oserror(monkeypatch):
    """OS-level read failure → WARN (defensive)."""
    monkeypatch.setattr(
        turn12_mod, "_count_ddm_schedule_variants",
        lambda _path: (False, "OSError: [Errno 2] No such file or directory"),
    )
    c = turn12_mod.ddm_continuous_schedule_variants_at_least_4()
    assert c.status == WARN
    assert "OSError" in c.details


def test_ddm_variants_passes_on_real_repo():
    """The real DDM oracle must expose >= 4 schedule variants.

    Real-on-disk run; no mocking. The turn-11
    ``ContinuousScheduleKind`` enum has exactly 4 variants
    (Linear, Cosine, Sqrt, Sigmoid). If this fails someone
    accidentally removed a variant.
    """
    c = turn12_mod.ddm_continuous_schedule_variants_at_least_4()
    assert c.status == PASS, (
        f"real DDM oracle did not expose 4 variants: {c.details}"
    )
    assert c.id == "ddm_continuous_schedule_variants_at_least_4"


# ---------------------------------------------------------------------------
# Both checks must be wired into the live CHECKS list (no orphans)
# ---------------------------------------------------------------------------


def test_both_turn12_checks_are_registered_in_doctor_checks():
    """Both check callables must be present in :data:`doctor.CHECKS`.

    Mirrors ``test_all_four_internal_checks_are_registered_in_doctor_checks``
    from the turn-10 test module.
    """
    from omlx_research.cli.doctor import CHECKS

    ids = {getattr(fn, "__name__", "") for fn in CHECKS}
    assert "cargo_workspace_crate_count_at_least_15" in ids
    assert "ddm_continuous_schedule_variants_at_least_4" in ids


# ---------------------------------------------------------------------------
# Live doctor subprocess: both new checks appear in the real report and the
# drift-detector threshold (28) accepts the new live count (28).
# ---------------------------------------------------------------------------


def test_live_doctor_includes_both_new_checks():
    """Spawn ``doctor --json`` for real and assert both new IDs are reported.

    Uses ``subprocess.run`` (per the task spec) so this exercises the
    full CLI surface — module import, subcommand dispatch, JSON
    serialization — rather than the in-process CHECKS list. The
    recursion-guard env var is unset for the parent; the spawned
    subprocess's own meta-check honors the guard and short-circuits.

    We grep the human-readable summary (not the JSON envelope) for
    the check IDs to match the task spec wording: "use subprocess.run
    ... and grep".
    """
    project_root = _doctor_project_root()
    # Run from the python/ source root so the subprocess's import
    # resolution matches the parent process.
    python_dir = os.path.join(project_root, "python")
    result = subprocess.run(
        [sys.executable, "-m", "omlx_research.cli", "doctor"],
        cwd=python_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Allow exit 0 (no warn/fail) or 1 (warn present, expected).
    assert result.returncode in (0, 1), (
        f"doctor subprocess exited {result.returncode}; "
        f"stderr={result.stderr[:200]!r}"
    )
    out = result.stdout
    assert "cargo_workspace_crate_count_at_least_15" in out, (
        f"new check id missing from doctor output:\n{out}"
    )
    assert "ddm_continuous_schedule_variants_at_least_4" in out, (
        f"new check id missing from doctor output:\n{out}"
    )


def test_live_doctor_meta_check_accepts_new_count():
    """The drift detector must see 28 checks and PASS.

    Spawns ``doctor --json`` and parses the JSON envelope to count
    rows + introspect the meta-check row's status. After schema-v2
    instrumented envelope gate the meta-check must:

    - observe ``len(envelope["checks"]) == 28`` (every registered
      check, including turn-12, FR-5, and schema-v2 adds);
    - report PASS with the threshold loaded from ``doctor_config.toml``
      as 28.
    """
    project_root = _doctor_project_root()
    python_dir = os.path.join(project_root, "python")
    result = subprocess.run(
        [sys.executable, "-m", "omlx_research.cli", "doctor", "--json"],
        cwd=python_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode in (0, 1)
    try:
        envelope = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(
            f"doctor --json produced invalid JSON: {e}; "
            f"stdout[:200]={result.stdout[:200]!r}"
        ) from e
    assert isinstance(envelope, dict) and "checks" in envelope
    assert len(envelope["checks"]) == 28, (
        f"expected 28 live doctor checks after schema-v2 envelope gate, got "
        f"{len(envelope['checks'])}"
    )
    ids = {c["id"] for c in envelope["checks"]}
    assert "cargo_workspace_crate_count_at_least_15" in ids
    assert "ddm_continuous_schedule_variants_at_least_4" in ids
    assert "julia_required_on_eval_path" in ids
    assert "niah_benchmark_non_legacy_path" in ids
    assert "niah_instrumented_schema_v2_present" in ids
    meta = next(
        (c for c in envelope["checks"] if c["id"] == "doctor_check_count_at_least_18"),
        None,
    )
    assert meta is not None, "meta-check missing from live envelope"
    assert meta["status"] == PASS, (
        f"meta-check did not PASS against the new threshold of 28: "
        f"{meta['details']!r}"
    )
    # The bumped threshold (28) must show up in the meta-check's
    # description/details. We check both the description and details
    # since either may carry the number depending on the path that
    # produced the line.
    meta_text = (meta.get("description", "") + " " + meta.get("details", ""))
    assert "28" in meta_text, (
        f"threshold '28' not surfaced in meta-check output: "
        f"{meta_text!r}"
    )


# ---------------------------------------------------------------------------
# Threshold-bump doc-comment in doctor_config.toml (lockstep rule)
# ---------------------------------------------------------------------------


def test_doctor_config_has_threshold_lockstep_doc_comment():
    """``doctor_config.toml`` must explain the lockstep rule.

    The check relies on a human reading the TOML understanding that
    any new mandatory check must bump ``min_check_count`` in
    lockstep. This test pins that the doc-comment is present so a
    future turn can't silently strip the explanation.
    """
    project_root = _doctor_project_root()
    config_path = os.path.join(
        project_root, "python", "omlx_research", "cli", "doctor_config.toml",
    )
    assert os.path.isfile(config_path), (
        f"doctor_config.toml missing at {config_path}"
    )
    with open(config_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    # The lockstep rule wording we pin (case-insensitive; allow for
    # any plausible phrasing the maintainer might choose):
    assert "lockstep" in text.lower(), (
        f"doctor_config.toml has no `LOCKSTEP` doc comment:\n{text}"
    )
    # And the threshold itself must currently be at 28 (schema-v2 gate):
    assert "min_check_count = 28" in text, (
        f"doctor_config.toml::min_check_count has not been bumped "
        f"to 28 (live count); got:\n{text}"
    )
