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
APPROVED_MODEL = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"


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


def _environment_env(job: str) -> dict[str, str]:
    """Read the known ``environment.env`` scalar mapping without a YAML dependency."""
    values: dict[str, str] = {}
    in_environment = False
    in_env = False
    for raw_line in job.splitlines():
        if not raw_line or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        stripped = raw_line.strip()
        if indent == 0 and stripped == "environment:":
            in_environment = True
            in_env = False
            continue
        if indent == 0:
            in_environment = False
            in_env = False
            continue
        if in_environment and indent == 2 and stripped == "env:":
            in_env = True
            continue
        if in_environment and indent == 2:
            in_env = False
            continue
        if not in_env or indent != 4 or ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        value = raw_value.strip().split(" #", 1)[0].strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        values[key] = value
    return values


def _endpoint_port(endpoint: str) -> int:
    match = re.fullmatch(r"http://host\.docker\.internal:(\d+)/v1", endpoint)
    if match is None:
        raise PreflightError("invalid local adapter endpoint")
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
    endpoint: str | None = None,
    openai_model: str | None = None,
    ready_model: str | None = None,
) -> LaunchPlan:
    """Validate a job without creating processes, network calls, or model loads."""
    job = job_path.read_text(encoding="utf-8") if job_text is None else job_text
    environment = _environment_env(job)
    model = environment.get("OPENAI_MODEL")
    job_ready_model = environment.get("OMLX_READY_MODEL")
    job_endpoint = environment.get("OPENAI_BASE_URL")
    if model is None or job_ready_model is None or job_endpoint is None:
        raise PreflightError("missing environment.env Qwen3.5 contract")
    effective_openai_model = model if openai_model is None else openai_model
    effective_ready_model = job_ready_model if ready_model is None else ready_model
    if (
        model != APPROVED_MODEL
        or job_ready_model != APPROVED_MODEL
        or effective_openai_model != APPROVED_MODEL
        or effective_ready_model != APPROVED_MODEL
    ):
        raise PreflightError("approved Qwen3.5 model required")
    if _int_value(job, "n_attempts") != 1:
        raise PreflightError("n_attempts must be 1")
    if _int_value(job, "n_concurrent_trials") != 1:
        raise PreflightError("n_concurrent_trials must be 1")
    if _int_value(job, "max_retries") != 0:
        raise PreflightError("max_retries must be 0")
    if available_memory_bytes < MIN_AVAILABLE_MEMORY_BYTES:
        raise PreflightError("available memory is below 4 GiB")
    declared_port = _endpoint_port(job_endpoint)
    if endpoint is not None and endpoint != job_endpoint:
        raise PreflightError("endpoint mismatch between job and launcher")
    if port is not None and port != declared_port:
        raise PreflightError(
            f"port mismatch: job declares {declared_port}, launcher requested {port}"
        )
    selected_port = declared_port
    if not port_available(selected_port):
        raise PreflightError(f"port {selected_port} is unavailable")
    return LaunchPlan(model=model, port=selected_port)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--openai-model", required=True)
    parser.add_argument("--ready-model", required=True)
    args = parser.parse_args()
    try:
        plan = preflight(
            args.job,
            available_memory_bytes=_available_memory_bytes(),
            port_available=_port_available,
            port=args.port,
            endpoint=args.endpoint,
            openai_model=args.openai_model,
            ready_model=args.ready_model,
        )
    except PreflightError as error:
        print(f"Qwen3.5 NIAH preflight: {error}", file=sys.stderr)
        return 2
    print(f"Qwen3.5 NIAH preflight: admitted model={plan.model} port={plan.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
