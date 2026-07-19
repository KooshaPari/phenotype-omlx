"""Tests for the doctor meta-check (turn-7 drift detector).

The meta-check :func:`omlx_research.cli._doctor_meta_checks.doctor_check_count_at_least_18`
spawns ``python -m omlx_research.cli doctor --json`` as a subprocess
and asserts ``len(envelope["checks"]) >= 18``. The threshold ladder is:

- ``count >= 18`` → PASS
- ``12 <= count < 18`` → WARN
- ``count < 12`` → FAIL

Every test in this module patches :mod:`subprocess.run` (via
:mod:`unittest.mock`) so the assertion runs in isolation — no
subprocess is actually spawned. The recursion guard is also
exercised.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

from omlx_research.cli import _doctor_meta_checks as meta_mod
from omlx_research.cli._doctor_shared import FAIL, PASS, WARN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _envelope(checks: list[dict]) -> dict:
    """Build a realistic ``doctor --json`` envelope around a check list."""
    return {
        "schema_version": 1,
        "ok": all(c.get("status") == "pass" for c in checks),
        "python_version": "3.14.0",
        "platform": "test",
        "checks": checks,
    }


def _check_entry(cid: str, status: str = "pass") -> dict:
    """Build one ``Check.asdict()`` row."""
    return {
        "id": cid,
        "description": cid,
        "status": status,
        "details": "",
    }


def _completed_process(envelope: dict, returncode: int = 0) -> MagicMock:
    """Build a fake ``subprocess.CompletedProcess`` carrying the envelope."""
    cp = MagicMock()
    cp.returncode = returncode
    cp.stdout = json.dumps(envelope)
    cp.stderr = ""
    return cp


def _ensure_env_unset(monkeypatch) -> None:
    """Make sure the recursion-guard env var is not leaking into the test."""
    monkeypatch.delenv(meta_mod._META_DEPTH_ENV, raising=False)


# ---------------------------------------------------------------------------
# Threshold ladder — pass branch
# ---------------------------------------------------------------------------


def test_threshold_pass_when_count_equals_floor(monkeypatch):
    """count == 18 → PASS (boundary: >= 18)."""
    _ensure_env_unset(monkeypatch)
    envelope = _envelope([_check_entry(f"c{i}") for i in range(18)])
    with patch.object(
        meta_mod.subprocess, "run", return_value=_completed_process(envelope),
    ):
        c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == PASS
    assert c.id == "doctor_check_count_at_least_18"
    assert "18 check(s)" in c.details


def test_threshold_pass_when_count_above_floor(monkeypatch):
    """count > 18 → PASS (the live registry currently reports 19)."""
    _ensure_env_unset(monkeypatch)
    envelope = _envelope([_check_entry(f"c{i}") for i in range(19)])
    with patch.object(
        meta_mod.subprocess, "run", return_value=_completed_process(envelope),
    ):
        c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == PASS
    assert "19 check(s)" in c.details


# ---------------------------------------------------------------------------
# Threshold ladder — warn branch
# ---------------------------------------------------------------------------


def test_threshold_warn_when_count_in_warn_band(monkeypatch):
    """count == 15 (between 12 and 17) → WARN."""
    _ensure_env_unset(monkeypatch)
    envelope = _envelope([_check_entry(f"c{i}") for i in range(15)])
    with patch.object(
        meta_mod.subprocess, "run", return_value=_completed_process(envelope),
    ):
        c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == WARN
    assert "15 check(s)" in c.details


def test_threshold_warn_at_lower_boundary(monkeypatch):
    """count == 12 → WARN (boundary: >= 12 and < 18)."""
    _ensure_env_unset(monkeypatch)
    envelope = _envelope([_check_entry(f"c{i}") for i in range(12)])
    with patch.object(
        meta_mod.subprocess, "run", return_value=_completed_process(envelope),
    ):
        c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == WARN
    assert "12 check(s)" in c.details


def test_threshold_warn_at_upper_boundary(monkeypatch):
    """count == 17 → WARN (boundary: < 18)."""
    _ensure_env_unset(monkeypatch)
    envelope = _envelope([_check_entry(f"c{i}") for i in range(17)])
    with patch.object(
        meta_mod.subprocess, "run", return_value=_completed_process(envelope),
    ):
        c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == WARN


# ---------------------------------------------------------------------------
# Threshold ladder — fail branch
# ---------------------------------------------------------------------------


def test_threshold_fail_when_count_below_floor(monkeypatch):
    """count == 8 (< 12) → FAIL."""
    _ensure_env_unset(monkeypatch)
    envelope = _envelope([_check_entry(f"c{i}") for i in range(8)])
    with patch.object(
        meta_mod.subprocess, "run", return_value=_completed_process(envelope),
    ):
        c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == FAIL
    assert "8 check(s)" in c.details


def test_threshold_fail_when_count_zero(monkeypatch):
    """count == 0 → FAIL."""
    _ensure_env_unset(monkeypatch)
    envelope = _envelope([])
    with patch.object(
        meta_mod.subprocess, "run", return_value=_completed_process(envelope),
    ):
        c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == FAIL


# ---------------------------------------------------------------------------
# Recursion guard
# ---------------------------------------------------------------------------


def test_short_circuits_when_meta_depth_env_set(monkeypatch):
    """When :data:`_META_DEPTH_ENV` is set, the meta-check returns PASS
    without spawning a subprocess — this is what breaks the recursion.
    """
    monkeypatch.setenv(meta_mod._META_DEPTH_ENV, "1")
    with patch.object(meta_mod.subprocess, "run") as mock_run:
        c = meta_mod.doctor_check_count_at_least_18()
    mock_run.assert_not_called()
    assert c.status == PASS
    assert "short-circuited" in c.details
    assert meta_mod._META_DEPTH_ENV in c.details


def test_short_circuits_even_with_empty_check_registry(monkeypatch):
    """The recursion guard must short-circuit regardless of the live
    count — otherwise a child subprocess would still spawn another.
    """
    monkeypatch.setenv(meta_mod._META_DEPTH_ENV, "1")
    with patch.object(meta_mod.subprocess, "run") as mock_run:
        c = meta_mod.doctor_check_count_at_least_18()
    mock_run.assert_not_called()
    assert c.status == PASS


# ---------------------------------------------------------------------------
# Subprocess failure modes → WARN
# ---------------------------------------------------------------------------


def test_subprocess_non_zero_exit_maps_to_warn(monkeypatch):
    """Non-zero/non-one exit from `doctor --json` → WARN."""
    _ensure_env_unset(monkeypatch)
    cp = _completed_process({}, returncode=2)
    cp.stderr = "boom"
    with patch.object(meta_mod.subprocess, "run", return_value=cp):
        c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == WARN
    assert "exited 2" in c.details


def test_subprocess_invalid_json_maps_to_warn(monkeypatch):
    """Invalid JSON on stdout → WARN."""
    _ensure_env_unset(monkeypatch)
    cp = MagicMock()
    cp.returncode = 0
    cp.stdout = "not json at all"
    cp.stderr = ""
    with patch.object(meta_mod.subprocess, "run", return_value=cp):
        c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == WARN
    assert "invalid JSON" in c.details


def test_subprocess_missing_checks_key_maps_to_warn(monkeypatch):
    """JSON envelope without a 'checks' key → WARN."""
    _ensure_env_unset(monkeypatch)
    envelope = {"schema_version": 1, "ok": True}
    with patch.object(
        meta_mod.subprocess, "run", return_value=_completed_process(envelope),
    ):
        c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == WARN
    assert "missing 'checks' key" in c.details


def test_subprocess_envelope_not_a_dict_maps_to_warn(monkeypatch):
    """JSON envelope whose root is not a dict → WARN."""
    _ensure_env_unset(monkeypatch)
    cp = MagicMock()
    cp.returncode = 0
    cp.stdout = "[1, 2, 3]"
    cp.stderr = ""
    with patch.object(meta_mod.subprocess, "run", return_value=cp):
        c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == WARN
    assert "missing 'checks' key" in c.details


def test_subprocess_timeout_maps_to_warn(monkeypatch):
    """Subprocess timeout → WARN (the check cannot conclude)."""
    _ensure_env_unset(monkeypatch)
    with patch.object(
        meta_mod.subprocess, "run",
        side_effect=subprocess.TimeoutExpired("cmd", 120),
    ):
        c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == WARN
    assert "TimeoutExpired" in c.details or "could not spawn" in c.details


def test_subprocess_oserror_maps_to_warn(monkeypatch):
    """OS-level failure to spawn the subprocess → WARN."""
    _ensure_env_unset(monkeypatch)
    with patch.object(
        meta_mod.subprocess, "run", side_effect=OSError("no such executable"),
    ):
        c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == WARN
    assert "OSError" in c.details


# ---------------------------------------------------------------------------
# Subprocess invocation shape
# ---------------------------------------------------------------------------


def test_subprocess_invocation_passes_meta_depth_env(monkeypatch):
    """The meta-check must set :data:`_META_DEPTH_ENV` in the child env
    so the child's nested meta-check short-circuits.
    """
    _ensure_env_unset(monkeypatch)
    envelope = _envelope([_check_entry(f"c{i}") for i in range(19)])
    with patch.object(
        meta_mod.subprocess, "run",
        return_value=_completed_process(envelope),
    ) as mock_run:
        meta_mod.doctor_check_count_at_least_18()
    args, kwargs = mock_run.call_args
    env = kwargs.get("env") or (args[0] if args else None)
    # The env must carry the recursion-guard variable.
    assert env is not None, "subprocess.run was not called with an env"
    assert env.get(meta_mod._META_DEPTH_ENV) == "1", (
        f"expected {meta_mod._META_DEPTH_ENV}='1' in child env, got {env}"
    )


def test_subprocess_invocation_uses_module_form(monkeypatch):
    """The command must be ``python -m omlx_research.cli doctor --json``
    so the spawn is independent of cwd / installed scripts.
    """
    _ensure_env_unset(monkeypatch)
    envelope = _envelope([_check_entry(f"c{i}") for i in range(19)])
    with patch.object(
        meta_mod.subprocess, "run",
        return_value=_completed_process(envelope),
    ) as mock_run:
        meta_mod.doctor_check_count_at_least_18()
    args, kwargs = mock_run.call_args
    cmd = args[0] if args else kwargs.get("args")
    assert cmd[:3] == ["python", "-m", "omlx_research.cli"] or (
        # sys.executable is used in place of "python"
        cmd[1] == "-m" and cmd[2] == "omlx_research.cli"
    ), f"unexpected subprocess cmd: {cmd!r}"
    assert "doctor" in cmd
    assert "--json" in cmd


# ---------------------------------------------------------------------------
# Live (non-mocked) sanity check — confirms the meta-check is wired in
# ---------------------------------------------------------------------------


def test_meta_check_is_registered_in_doctor_checks():
    """The meta-check function must be the last entry in
    :data:`omlx_research.cli.doctor.CHECKS` so it observes the complete
    registry. This is a wiring test, not a behavior test.
    """
    from omlx_research.cli.doctor import CHECKS

    assert CHECKS[-1] is meta_mod.doctor_check_count_at_least_18, (
        "meta-check must be the last entry in CHECKS so the count it "
        "observes reflects the complete registry"
    )


def test_meta_check_passes_on_real_registry():
    """Real run: the meta-check must PASS against the live registry
    (which contains the meta-check itself, so count == 19 >= 18).

    This test invokes the real subprocess, so it is the only one in
    this module that does not mock subprocess.run. It exercises the
    recursion guard end-to-end.
    """
    c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == PASS
    assert c.id == "doctor_check_count_at_least_18"
    assert "check(s)" in c.details