"""Fail-closed, non-executing admission for the local Qwen3.5 NIAH Harbor job."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

MIN_AVAILABLE_MEMORY_BYTES = 4 * 1024**3


class PreflightError(ValueError):
    """The local job is unsafe to launch."""


@dataclass(frozen=True)
class LaunchPlan:
    """An admitted plan, explicitly without workload side effects."""

    model: str
    port: int
    workload_executed: bool = False
    model_loaded: bool = False


def _value(job: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}:\s*[\"']?([^\"'#\n]+)", job, re.MULTILINE)
    if match is None:
        raise PreflightError(f"missing {key}")
    return match.group(1).strip()


def _int_value(job: str, key: str) -> int:
    try:
        return int(_value(job, key))
    except ValueError as error:
        raise PreflightError(f"invalid {key}") from error


def _port(job: str) -> int:
    match = re.search(r"http://[^/:]+:(\d+)(?:/|\")", job)
    if match is None:
        raise PreflightError("missing local adapter port")
    return int(match.group(1))


def preflight(
    job_path: Path,
    *,
    available_memory_bytes: int,
    port_available: Callable[[int], bool],
    job_text: str | None = None,
) -> LaunchPlan:
    """Validate a job without creating processes, network calls, or model loads."""
    job = job_path.read_text(encoding="utf-8") if job_text is None else job_text
    model = _value(job, "OPENAI_MODEL")
    if "Qwen3.5" not in model:
        raise PreflightError("Qwen3.5-only model required")
    if _int_value(job, "n_attempts") != 1:
        raise PreflightError("n_attempts must be 1")
    if _int_value(job, "n_concurrent_trials") != 1:
        raise PreflightError("n_concurrent_trials must be 1")
    if _int_value(job, "max_retries") != 0:
        raise PreflightError("max_retries must be 0")
    if available_memory_bytes < MIN_AVAILABLE_MEMORY_BYTES:
        raise PreflightError("available memory is below 4 GiB")
    port = _port(job)
    if not port_available(port):
        raise PreflightError(f"port {port} is unavailable")
    return LaunchPlan(model=model, port=port)
