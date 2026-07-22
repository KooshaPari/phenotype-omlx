"""Isolated one-token Lite-probe boundary for real-model evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

if "e2e_validation" in sys.modules:
    validation = sys.modules["e2e_validation"]
elif "scripts.e2e_validation" in sys.modules:
    validation = sys.modules["scripts.e2e_validation"]
else:
    try:
        import e2e_validation as validation
    except ModuleNotFoundError:
        from scripts import e2e_validation as validation

try:
    from e2e_real_model_workload import BenchmarkWorkload
except ModuleNotFoundError:
    from scripts.e2e_real_model_workload import BenchmarkWorkload


MAX_PROBE_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class LiteProbeResult:
    """Ephemeral capability evidence; deliberately contains no publish path."""

    saved_bytes: int
    baseline_logits_finite: bool
    compacted_logits_finite: bool


def run_one_token_lite_probe(
    *,
    workload: BenchmarkWorkload,
    stock_cache_factory: Callable[[], object],
    lite_cache_factory: Callable[[], object],
    prefill: Callable[[object, str], object],
    compact: Callable[[object], int],
    score_one_token: Callable[[object, int], object],
    first_token_id: Callable[[str], Sequence[object]],
    logits_are_finite: Callable[[object], bool],
    bounded_execute: Callable[[str, Callable[[], object], float], object],
    timeout_seconds: float,
) -> LiteProbeResult:
    """Probe one post-compaction token without emitting validation evidence."""

    if timeout_seconds <= 0:
        raise validation.ValidationError("probe timeout must be positive")
    token_ids = tuple(first_token_id(workload.teacher_forced_continuation))
    if (
        len(token_ids) != 1
        or not isinstance(token_ids[0], int)
        or isinstance(token_ids[0], bool)
    ):
        raise validation.ValidationError("probe requires exactly one integer token ID")
    token_id = token_ids[0]
    stock_cache = stock_cache_factory()
    bounded_execute("stock-prefill", lambda: prefill(stock_cache, workload.prompt), timeout_seconds)
    baseline_logits = bounded_execute(
        "stock-score", lambda: score_one_token(stock_cache, token_id), timeout_seconds
    )
    lite_cache = lite_cache_factory()
    bounded_execute("lite-prefill", lambda: prefill(lite_cache, workload.prompt), timeout_seconds)
    saved_bytes = bounded_execute("lite-compact", lambda: compact(lite_cache), timeout_seconds)
    if not isinstance(saved_bytes, int) or isinstance(saved_bytes, bool) or saved_bytes <= 0:
        raise validation.ValidationError("Lite probe compaction must save nonzero bytes")
    compacted_logits = bounded_execute(
        "lite-score", lambda: score_one_token(lite_cache, token_id), timeout_seconds
    )
    result = LiteProbeResult(
        saved_bytes=saved_bytes,
        baseline_logits_finite=bool(logits_are_finite(baseline_logits)),
        compacted_logits_finite=bool(logits_are_finite(compacted_logits)),
    )
    if not result.baseline_logits_finite or not result.compacted_logits_finite:
        raise validation.ValidationError("Lite probe logits must be finite")
    return result


def run_bounded_probe_process(
    *,
    command: Sequence[str],
    request: Mapping[str, str],
    timeout_seconds: float,
) -> dict[str, object]:
    """Run an isolated probe child with JSON-only immutable inputs and timeout."""

    if (
        not command
        or not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= MAX_PROBE_TIMEOUT_SECONDS
    ):
        raise validation.ValidationError(
            f"probe timeout must be finite and within 0..{MAX_PROBE_TIMEOUT_SECONDS:g} seconds"
        )
    revisions = {name: request.get(name) for name in ("model_revision", "workload_revision")}
    if any(not isinstance(revision, str) or not revision for revision in revisions.values()):
        raise validation.ValidationError("probe request must include immutable model and workload revisions")
    approved_child = Path(__file__).with_name("lite_probe_child.py").resolve()
    approved_python = Path(sys.executable).resolve()
    if len(command) != 2:
        raise validation.ValidationError("probe command must invoke the approved lite_probe_child.py")
    try:
        command_python = Path(command[0]).resolve(strict=True)
        command_child = Path(command[1]).resolve(strict=True)
    except (OSError, TypeError) as error:
        raise validation.ValidationError("probe command must contain resolvable approved executables") from error
    if command_python != approved_python:
        raise validation.ValidationError("probe command must invoke the approved Python interpreter")
    if command_child != approved_child:
        raise validation.ValidationError("probe command must invoke the approved lite_probe_child.py")
    payload = json.dumps(dict(request), sort_keys=True, allow_nan=False)
    environment = {"PATH": os.defpath, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=environment, start_new_session=True,
    )
    try:
        stdout, _stderr = process.communicate(payload, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise TimeoutError("isolated Lite probe process exceeded its timeout") from error
    if process.returncode != 0:
        raise RuntimeError(f"isolated Lite probe failed with exit {process.returncode}")
    try:
        response = json.loads(stdout)
    except ValueError as error:
        raise validation.ValidationError("isolated Lite probe returned invalid JSON") from error
    if not isinstance(response, dict):
        raise validation.ValidationError("isolated Lite probe response must be an object")
    if response.get("publication") is not False or response.get("status") != "capability_pending":
        raise validation.ValidationError("isolated Lite probe response violates capability schema")
    if any(response.get(name) != revision for name, revision in revisions.items()):
        raise validation.ValidationError("isolated Lite probe response does not bind immutable request revisions")
    return response
