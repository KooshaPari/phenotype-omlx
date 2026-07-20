"""Doctor meta-checks added 2026-07-19 — turn 7 batch.

This module holds the meta-check that asserts the live doctor check
count meets a minimum threshold. It catches doctor-drift: if someone
deletes a check from :data:`omlx_research.cli.doctor.CHECKS`, the
meta-check flips to WARN (count between the WARN floor and the PASS
threshold) or FAIL (count below the WARN floor).

Implementation
--------------
The meta-check spawns ``python -m omlx_research.cli doctor --json`` as
a subprocess and parses the resulting JSON envelope. The ``checks``
list length in that envelope is the count we assert against.

Recursion guard
---------------
Without a guard, the subprocess's own doctor would include the
meta-check in its ``CHECKS`` list and would itself spawn another
subprocess, ad infinitum. To break this cycle the meta-check honors
the internal environment variable :data:`_META_DEPTH_ENV`: when set
in the subprocess's environment, the meta-check short-circuits to
PASS without spawning another subprocess. The outermost invocation
sees the env var unset and runs the full subprocess + parse cycle.

Threshold ladder
----------------
The PASS threshold is configurable via the sibling TOML file
``doctor_config.toml`` under ``[meta].min_check_count``. When the
file is missing, malformed, or the key is absent, the meta-check
falls back to :data:`_DEFAULT_MIN_CHECK_COUNT` (= 18).

The threshold ladder is intentionally generous so a single accidental
deletion does not immediately break CI:

- ``count >= threshold`` → PASS
- ``_THRESHOLD_FAIL <= count < threshold`` → WARN
- ``count < _THRESHOLD_FAIL`` → FAIL
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    # Python 3.11+ stdlib
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover — Python <3.11 fallback
    import tomli as tomllib  # type: ignore[import-untyped,no-redef]

from ._doctor_shared import FAIL, PASS, WARN, Check


__all__ = ["doctor_check_count_at_least_18"]


#: Environment variable used to break the recursion. When set to a
#: truthy value in the current process environment, the meta-check
#: short-circuits to PASS without spawning a subprocess. The meta-check
#: itself sets this in the child environment when it spawns the
#: ``doctor --json`` subprocess.
_META_DEPTH_ENV = "OMLX_DOCTOR_META_DEPTH"

#: Default PASS threshold when no config file is found. The current
#: live registry is 19 checks (18 base + this meta-check), so the
#: default of 18 preserves the prior hard-coded behavior when the
#: config is missing, malformed, or lacks the key.
_DEFAULT_MIN_CHECK_COUNT: int = 18

#: Filename (sibling of this module) that holds the configured PASS
#: threshold under ``[meta].min_check_count``. Loaded by
#: :func:`_load_min_check_count`.
_CONFIG_FILENAME: str = "doctor_config.toml"

#: Number of doctor checks below which the meta-check FAILs. The
#: band between :data:`_THRESHOLD_FAIL` (inclusive) and
#: :data:`_THRESHOLD_PASS` (exclusive) is WARN — a drift signal that
#: demands attention but is not yet a hard break.
_THRESHOLD_FAIL: int = 12

#: Subprocess timeout (seconds). The doctor itself can take several
#: seconds when ``tests_runnable`` runs ``pytest --collect-only``, so
#: the timeout is generous.
_SUBPROCESS_TIMEOUT_SECONDS: int = 120


def _python_source_root() -> str:
    """Locate the on-disk source root for the ``omlx_research`` package.

    The meta-check lives at ``python/omlx_research/cli/_doctor_meta_checks.py``,
    so the source root is the parent of the ``omlx_research`` directory — that
    is ``<repo>/python``. We compute it from :data:`__file__` rather than from
    :data:`sys.path` because the source root is needed in the *child*
    environment of the subprocess, not the parent's import path.

    Returns the absolute path as a string. Falls back to an empty string if
    the module is loaded from a zipapp or other location where the path
    arithmetic cannot be performed; the caller treats an empty string as
    "PYTHONPATH not augmented" which is correct in pip-installed environments.
    """
    try:
        # .../_doctor_meta_checks.py  →  .../cli  →  .../omlx_research  →  .../python
        return str(Path(__file__).resolve().parent.parent.parent)
    except (OSError, ValueError):
        return ""


def _config_path() -> Path | None:
    """Locate :data:`_CONFIG_FILENAME` as a sibling of this module.

    Returns the resolved :class:`pathlib.Path` when the file exists on
    disk, ``None`` otherwise. ``None`` is also returned for any path
    resolution failure (e.g. loaded from a zipapp where ``__file__``
    is not on disk). The caller treats ``None`` as "no config, use
    default" — a silent degradation that keeps the doctor running.
    """
    try:
        candidate = (Path(__file__).resolve().parent / _CONFIG_FILENAME)
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() else None


def _load_min_check_count(default: int = _DEFAULT_MIN_CHECK_COUNT) -> int:
    """Read the PASS threshold from :data:`_CONFIG_FILENAME`.

    The config file is a TOML document sibling of this module. The
    relevant key is ``[meta].min_check_count`` — an integer that the
    meta-check compares against the live doctor check count.

    Fallback policy (silent — no logging):

    - Missing config file → ``default``
    - Malformed TOML → ``default``
    - Missing ``[meta]`` table → ``default``
    - Missing ``min_check_count`` key → ``default``
    - Non-integer value → ``default``

    The default argument is exposed so tests can inject a synthetic
    baseline without touching the real default.
    """
    path = _config_path()
    if path is None:
        return default
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return default
    if not isinstance(data, dict):
        return default
    meta = data.get("meta")
    if not isinstance(meta, dict):
        return default
    value = meta.get("min_check_count", default)
    if isinstance(value, bool) or not isinstance(value, int):
        # ``bool`` is a subclass of ``int`` in Python; reject it so
        # ``min_check_count = true`` does not silently mean ``1``.
        return default
    return value


def _run_doctor_json_subprocess() -> dict[str, Any]:
    """Spawn ``doctor --json`` and return the parsed JSON envelope.

    Sets :data:`_META_DEPTH_ENV` in the child environment so the
    subprocess's nested meta-check short-circuits instead of recursing
    into another subprocess. Also injects the on-disk ``python/``
    source root into ``PYTHONPATH`` so the child can import
    ``omlx_research`` even when the package is not pip-installed in the
    user's environment (e.g. when running ``pytest`` directly from a
    source checkout via rootdir auto-discovery).

    Raises ``RuntimeError`` on non-zero/non-one exit, missing/invalid
    JSON, or a missing ``checks`` key — the caller maps these to WARN.
    Raises ``OSError`` / :class:`subprocess.TimeoutExpired` if the
    subprocess cannot be started or times out — the caller maps these
    to WARN as well.
    """
    env = os.environ.copy()
    env[_META_DEPTH_ENV] = "1"
    src_root = _python_source_root()
    if src_root:
        # Prepend so user-set PYTHONPATH entries still win for shadowing,
        # but the on-disk source root is found before stdlib lookups.
        env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "omlx_research.cli", "doctor", "--json"],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        env=env,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"doctor --json exited {proc.returncode}: "
            f"{(proc.stderr or '').strip()[:200]}"
        )
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"doctor --json produced invalid JSON: {e}: "
            f"stdout[:200]={proc.stdout[:200]!r}"
        ) from e
    if not isinstance(envelope, dict) or "checks" not in envelope:
        raise RuntimeError(
            f"doctor --json envelope missing 'checks' key: {envelope!r}"
        )
    return envelope


def doctor_check_count_at_least_18() -> Check:
    """Assert the live doctor check count meets the PASS threshold.

    Drift detector: if someone deletes one or more checks from
    :data:`omlx_research.cli.doctor.CHECKS`, this meta-check will
    flip to WARN (count in ``[_THRESHOLD_FAIL, threshold)``) or
    FAIL (count below :data:`_THRESHOLD_FAIL`).

    The PASS threshold is read at call time from
    :data:`_CONFIG_FILENAME` (sibling TOML) under
    ``[meta].min_check_count``. When the file is missing, malformed,
    or the key is absent, the meta-check falls back to
    :data:`_DEFAULT_MIN_CHECK_COUNT` (= 18).

    Implementation:

    - When :data:`_META_DEPTH_ENV` is set in the current environment,
      short-circuit to PASS. This breaks the recursion: the spawned
      subprocess's nested meta-check must not itself spawn a
      subprocess.
    - Otherwise, spawn ``python -m omlx_research.cli doctor --json``
      with :data:`_META_DEPTH_ENV` set in the child env, parse the
      resulting JSON envelope, and count ``envelope['checks']``.
    """
    threshold_pass = _load_min_check_count()
    desc = (
        f"Doctor check count >= {threshold_pass} "
        f"(drift detector; threshold from {_CONFIG_FILENAME}; "
        f"fail < {_THRESHOLD_FAIL}, warn < {threshold_pass})"
    )

    # Recursion guard: in a subprocess, skip the spawn.
    if os.environ.get(_META_DEPTH_ENV):
        return Check(
            id="doctor_check_count_at_least_18",
            description=desc,
            status=PASS,
            details=(
                f"{_META_DEPTH_ENV}={os.environ[_META_DEPTH_ENV]!r} is set; "
                f"meta-check short-circuited to PASS to break recursion"
            ),
        )

    try:
        envelope = _run_doctor_json_subprocess()
    except (OSError, subprocess.TimeoutExpired) as e:
        return Check(
            id="doctor_check_count_at_least_18",
            description=desc,
            status=WARN,
            details=(
                f"could not spawn `python -m omlx_research.cli doctor "
                f"--json`: {type(e).__name__}: {e}"
            ),
        )
    except RuntimeError as e:
        return Check(
            id="doctor_check_count_at_least_18",
            description=desc,
            status=WARN,
            details=str(e),
        )

    checks = envelope.get("checks", [])
    count = len(checks) if isinstance(checks, list) else 0
    if count >= threshold_pass:
        status = PASS
    elif count >= _THRESHOLD_FAIL:
        status = WARN
    else:
        status = FAIL

    details = (
        f"live doctor reported {count} check(s); "
        f"thresholds: pass >= {threshold_pass} "
        f"(from {_CONFIG_FILENAME}), "
        f"warn >= {_THRESHOLD_FAIL}, fail < {_THRESHOLD_FAIL}"
    )
    return Check(
        id="doctor_check_count_at_least_18",
        description=desc,
        status=status,
        details=details,
    )