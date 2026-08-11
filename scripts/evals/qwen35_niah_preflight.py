"""Fail-closed, non-executing admission for the local Qwen3.5 NIAH Harbor job."""

from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
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


def _available_memory_bytes() -> int:
    """Return conservative macOS reclaimable memory or fail closed."""
    try:
        output = subprocess.run(
            ["/usr/bin/vm_stat"], text=True, capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightError("available memory observation is unavailable") from error
    page_size = re.search(r"page size of (\d+) bytes", output)
    if page_size is None:
        raise PreflightError("available memory observation is unavailable")
    pages = 0
    for name in ("Pages free", "Pages speculative", "Pages purgeable"):
        match = re.search(rf"^{re.escape(name)}:\s+(\d+)", output, re.MULTILINE)
        if match is None:
            raise PreflightError("available memory observation is unavailable")
        pages += int(match.group(1))
    return pages * int(page_size.group(1))


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


def preflight(
    job_path: Path,
    *,
    available_memory_bytes: int,
    port_available: Callable[[int], bool],
    job_text: str | None = None,
    port: int | None = None,
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
    selected_port = _port(job) if port is None else port
    if not port_available(selected_port):
        raise PreflightError(f"port {selected_port} is unavailable")
    return LaunchPlan(model=model, port=selected_port)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    try:
        plan = preflight(
            args.job,
            available_memory_bytes=_available_memory_bytes(),
            port_available=_port_available,
            port=args.port,
        )
    except PreflightError as error:
        print(f"Qwen3.5 NIAH preflight: {error}", file=sys.stderr)
        return 2
    print(f"Qwen3.5 NIAH preflight: admitted model={plan.model} port={plan.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
