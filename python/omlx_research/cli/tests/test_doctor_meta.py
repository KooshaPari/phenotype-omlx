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


def _patch_threshold(monkeypatch, value: int):
    """Force ``_load_min_check_count`` to return ``value`` for this test.

    The threshold is now config-driven (sibling ``doctor_config.toml``).
    The threshold-ladder tests in this module assert behavior at the
    *prior* hard-coded values (18 / 12), so we pin the loader to those
    values rather than letting the real config dictate the ladder.
    """
    monkeypatch.setattr(meta_mod, "_load_min_check_count", lambda default=18: value)


# ---------------------------------------------------------------------------
# Threshold ladder — pass branch
# ---------------------------------------------------------------------------


def test_threshold_pass_when_count_equals_floor(monkeypatch):
    """count == 18 → PASS (boundary: >= 18)."""
    _ensure_env_unset(monkeypatch)
    _patch_threshold(monkeypatch, 18)
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
    _patch_threshold(monkeypatch, 18)
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
    _patch_threshold(monkeypatch, 18)
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
    _patch_threshold(monkeypatch, 18)
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
    _patch_threshold(monkeypatch, 18)
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
    _patch_threshold(monkeypatch, 18)
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
    _patch_threshold(monkeypatch, 18)
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
# PYTHONPATH injection — turn-8 fix so live subprocess finds omlx_research
# ---------------------------------------------------------------------------


def test_python_source_root_is_repo_python_dir():
    """`_python_source_root` must return the absolute path to the
    repo's ``python/`` directory — three parents above this module
    (cli → omlx_research → python).
    """
    from pathlib import Path

    src = meta_mod._python_source_root()
    assert src, "_python_source_root returned an empty string"
    p = Path(src)
    assert p.is_absolute(), f"_python_source_root must be absolute, got {src!r}"
    assert (p / "omlx_research" / "cli" / "_doctor_meta_checks.py").exists(), (
        f"_python_source_root does not point to the repo python/ root: {src!r}"
    )


def test_subprocess_env_includes_python_source_root(monkeypatch):
    """The subprocess must receive PYTHONPATH containing the source
    root so the child interpreter can ``import omlx_research`` even
    when the package is not pip-installed.
    """
    _ensure_env_unset(monkeypatch)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    envelope = _envelope([_check_entry(f"c{i}") for i in range(19)])
    with patch.object(
        meta_mod.subprocess, "run",
        return_value=_completed_process(envelope),
    ) as mock_run:
        meta_mod.doctor_check_count_at_least_18()
    args, kwargs = mock_run.call_args
    env = kwargs.get("env") or args[0]
    assert env is not None, "subprocess.run was not called with env"
    pp = env.get("PYTHONPATH", "")
    src = meta_mod._python_source_root()
    assert src in pp, (
        f"PYTHONPATH must contain source root {src!r}, got {pp!r}"
    )


def test_subprocess_env_preserves_existing_pythonpath(monkeypatch):
    """An existing ``PYTHONPATH`` entry must be preserved (and the
    source root prepended, not replaced).
    """
    _ensure_env_unset(monkeypatch)
    sentinel = "/tmp/some-existing-pythonpath-entry"
    monkeypatch.setenv("PYTHONPATH", sentinel)
    envelope = _envelope([_check_entry(f"c{i}") for i in range(19)])
    with patch.object(
        meta_mod.subprocess, "run",
        return_value=_completed_process(envelope),
    ) as mock_run:
        meta_mod.doctor_check_count_at_least_18()
    args, kwargs = mock_run.call_args
    env = kwargs.get("env")
    pp = env.get("PYTHONPATH", "")
    src = meta_mod._python_source_root()
    assert sentinel in pp, f"existing PYTHONPATH entry lost: got {pp!r}"
    assert src in pp, f"source root missing from PYTHONPATH: got {pp!r}"


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


# ---------------------------------------------------------------------------
# Config-driven threshold (turn-8 forward priority #4)
# ---------------------------------------------------------------------------


def test_load_min_check_count_reads_from_sibling_toml(tmp_path, monkeypatch):
    """``_load_min_check_count`` returns the integer under
    ``[meta].min_check_count`` in the sibling ``doctor_config.toml``.
    """
    cfg = tmp_path / "doctor_config.toml"
    cfg.write_text('[meta]\nmin_check_count = 27\n')
    # Point the loader at the temp file by patching ``__file__`` to a
    # sibling-of-cfg marker. ``_config_path`` resolves via
    # ``Path(__file__).resolve().parent``, so we synthesize a fake
    # module file in tmp_path whose sibling is ``cfg``.
    fake_module = tmp_path / "_doctor_meta_checks.py"
    fake_module.write_text("# placeholder for path arithmetic\n")
    monkeypatch.setattr(meta_mod, "__file__", str(fake_module))
    assert meta_mod._load_min_check_count() == 27


def test_load_min_check_count_falls_back_when_config_missing(tmp_path, monkeypatch):
    """When the sibling TOML file does not exist, the loader returns the
    supplied ``default`` argument — silent degradation.
    """
    fake_module = tmp_path / "_doctor_meta_checks.py"
    fake_module.write_text("# placeholder for path arithmetic\n")
    # ``doctor_config.toml`` is intentionally NOT created.
    monkeypatch.setattr(meta_mod, "__file__", str(fake_module))
    assert meta_mod._load_min_check_count() == meta_mod._DEFAULT_MIN_CHECK_COUNT
    # Custom default is honored too.
    assert meta_mod._load_min_check_count(default=42) == 42


def test_load_min_check_count_falls_back_when_toml_malformed(tmp_path, monkeypatch):
    """When the TOML is malformed, the loader returns the default —
    it must never raise or log to the user.
    """
    fake_module = tmp_path / "_doctor_meta_checks.py"
    fake_module.write_text("# placeholder for path arithmetic\n")
    cfg = fake_module.parent / meta_mod._CONFIG_FILENAME
    cfg.write_text("this is = not [ valid toml")  # unmatched bracket
    monkeypatch.setattr(meta_mod, "__file__", str(fake_module))
    assert meta_mod._load_min_check_count() == meta_mod._DEFAULT_MIN_CHECK_COUNT


def test_load_min_check_count_falls_back_when_key_absent(tmp_path, monkeypatch):
    """When the file parses but ``[meta].min_check_count`` is missing,
    the loader returns the default.
    """
    fake_module = tmp_path / "_doctor_meta_checks.py"
    fake_module.write_text("# placeholder for path arithmetic\n")
    cfg = fake_module.parent / meta_mod._CONFIG_FILENAME
    cfg.write_text('[meta]\nsome_other_key = 1\n')  # no min_check_count
    monkeypatch.setattr(meta_mod, "__file__", str(fake_module))
    assert meta_mod._load_min_check_count() == meta_mod._DEFAULT_MIN_CHECK_COUNT


def test_threshold_from_config_flows_through_to_check_status(
    tmp_path, monkeypatch,
):
    """End-to-end: a config-driven threshold changes the meta-check's
    verdict. With threshold=30 and live count=19, the meta-check is
    WARN (19 in ``[_THRESHOLD_FAIL, threshold)``). With threshold=10
    and live count=19, the meta-check is PASS.

    This is the integration test that proves the refactor preserves
    the threshold-ladder contract while moving the knob to TOML.
    """
    _ensure_env_unset(monkeypatch)

    # Threshold = 30: count 19 is in the warn band.
    monkeypatch.setattr(meta_mod, "_load_min_check_count", lambda default=18: 30)
    envelope = _envelope([_check_entry(f"c{i}") for i in range(19)])
    with patch.object(
        meta_mod.subprocess, "run", return_value=_completed_process(envelope),
    ):
        c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == WARN, (
        f"with threshold=30 and count=19 expected WARN, got {c.status!r}"
    )
    assert "30" in c.description
    assert meta_mod._CONFIG_FILENAME in c.description

    # Threshold = 10: count 19 is comfortably above PASS.
    monkeypatch.setattr(meta_mod, "_load_min_check_count", lambda default=18: 10)
    with patch.object(
        meta_mod.subprocess, "run", return_value=_completed_process(envelope),
    ):
        c = meta_mod.doctor_check_count_at_least_18()
    assert c.status == PASS, (
        f"with threshold=10 and count=19 expected PASS, got {c.status!r}"
    )
    assert "10" in c.description