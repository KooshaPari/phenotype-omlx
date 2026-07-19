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
The threshold ladder is intentionally generous so a single accidental
deletion does not immediately break CI:

- ``count >= 18`` → PASS
- ``12 <= count < 18`` → WARN
- ``count < 12`` → FAIL
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from ._doctor_shared import FAIL, PASS, WARN, Check


__all__ = ["doctor_check_count_at_least_18"]


#: Environment variable used to break the recursion. When set to a
#: truthy value in the current process environment, the meta-check
#: short-circuits to PASS without spawning a subprocess. The meta-check
#: itself sets this in the child environment when it spawns the
#: ``doctor --json`` subprocess.
_META_DEPTH_ENV = "OMLX_DOCTOR_META_DEPTH"

#: Number of doctor checks at or above which the meta-check PASSes.
#: The number is derived from the live registry at the time this
#: meta-check was added: 18 base checks + this meta-check = 19, well
#: above the floor.
_THRESHOLD_PASS: int = 18

#: Number of doctor checks below which the meta-check FAILs. The
#: band between :data:`_THRESHOLD_FAIL` (inclusive) and
#: :data:`_THRESHOLD_PASS` (exclusive) is WARN — a drift signal that
#: demands attention but is not yet a hard break.
_THRESHOLD_FAIL: int = 12

#: Subprocess timeout (seconds). The doctor itself can take several
#: seconds when ``tests_runnable`` runs ``pytest --collect-only``, so
#: the timeout is generous.
_SUBPROCESS_TIMEOUT_SECONDS: int = 120


def _run_doctor_json_subprocess() -> dict[str, Any]:
    """Spawn ``doctor --json`` and return the parsed JSON envelope.

    Sets :data:`_META_DEPTH_ENV` in the child environment so the
    subprocess's nested meta-check short-circuits instead of recursing
    into another subprocess.

    Raises ``RuntimeError`` on non-zero/non-one exit, missing/invalid
    JSON, or a missing ``checks`` key — the caller maps these to WARN.
    Raises ``OSError`` / :class:`subprocess.TimeoutExpired` if the
    subprocess cannot be started or times out — the caller maps these
    to WARN as well.
    """
    env = os.environ.copy()
    env[_META_DEPTH_ENV] = "1"
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
    flip to WARN (count in ``[_THRESHOLD_FAIL, _THRESHOLD_PASS)``) or
    FAIL (count below :data:`_THRESHOLD_FAIL`).

    Implementation:

    - When :data:`_META_DEPTH_ENV` is set in the current environment,
      short-circuit to PASS. This breaks the recursion: the spawned
      subprocess's nested meta-check must not itself spawn a
      subprocess.
    - Otherwise, spawn ``python -m omlx_research.cli doctor --json``
      with :data:`_META_DEPTH_ENV` set in the child env, parse the
      resulting JSON envelope, and count ``envelope['checks']``.
    """
    desc = (
        f"doctor drift detector: live check count must be >= "
        f"{_THRESHOLD_PASS} (fail < {_THRESHOLD_FAIL}, "
        f"warn < {_THRESHOLD_PASS})"
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
    if count >= _THRESHOLD_PASS:
        status = PASS
    elif count >= _THRESHOLD_FAIL:
        status = WARN
    else:
        status = FAIL

    details = (
        f"live doctor reported {count} check(s); "
        f"thresholds: pass >= {_THRESHOLD_PASS}, "
        f"warn >= {_THRESHOLD_FAIL}, fail < {_THRESHOLD_FAIL}"
    )
    return Check(
        id="doctor_check_count_at_least_18",
        description=desc,
        status=status,
        details=details,
    )