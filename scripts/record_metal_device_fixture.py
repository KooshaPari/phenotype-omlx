#!/usr/bin/env python3
"""Run one bounded Metal fixture and record external, non-promotable evidence.

The recorder requires a current-head compile-only provenance record, an
allowlisted metallib, and one of the small ignored parity fixtures.  It never
loads a model, starts Harbor, or runs an evaluation workload.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import re
from typing import Any, Callable


FIXTURES = {
    "diffusion": ("diffusion_dispatch", "diffusion_three_stage_fixture_matches_oracle"),
    "ternary-small": ("ternary", "metal_matches_scalar_reference"),
    "ternary-edge": ("ternary", "metal_matches_edge_shape_with_all_packed_codes"),
}

_MIN_AVAILABLE_MEMORY_BYTES = 4 * 1024**3
_MAX_LOAD_PER_LOGICAL_CPU = 0.75


@dataclass(frozen=True)
class ResourceSnapshot:
    """Host state required before a bounded device-fixture dispatch."""

    logical_cpu_count: int
    load_average_1m: float
    available_memory_bytes: int
    source: str


ResourceObserver = Callable[[], ResourceSnapshot]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()


def _require_clean_head(repo_root: Path) -> tuple[str, str]:
    if _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("repository must be clean before fixture evidence capture")
    head = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "branch", "--show-current")
    if not branch:
        raise RuntimeError("repository must be on a named branch")
    return head, branch


def _load_compile_provenance(path: Path, current_head: str, artifact: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("compile provenance must be readable UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise RuntimeError("compile provenance root must be an object")
    if document.get("candidate_source_head") != current_head or document.get(
        "build_checkout_head"
    ) != current_head:
        raise RuntimeError("compile provenance is not bound to current HEAD")
    if document.get("status") != "current_head_compile_only":
        raise RuntimeError("compile provenance is not current-head compile-only evidence")
    if any(document.get(field) is not False for field in ("workload_executed", "model_loaded")):
        raise RuntimeError("compile provenance must not claim workload or model execution")
    artifact_digest = _sha256(artifact)
    if document.get("metallib_sha256") != artifact_digest:
        raise RuntimeError("artifact SHA-256 does not match compile provenance")
    return document


def _require_external_output(output: Path, repo_root: Path) -> Path:
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"output already exists: {output}")
    try:
        output.relative_to(repo_root)
    except ValueError:
        return output
    raise RuntimeError("fixture evidence output must be outside the repository")


def _positive_sysctl_u64(name: str) -> int:
    try:
        value = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL
        ).strip()
        parsed = int(value)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"resource governor observability unavailable: sysctl {name}") from exc
    if parsed < 1:
        raise RuntimeError(f"resource governor observability unavailable: sysctl {name}")
    return parsed


def _available_memory_from_vm_stat() -> int:
    try:
        output = subprocess.check_output(
            ["/usr/bin/vm_stat"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("resource governor observability unavailable: vm_stat") from exc
    page_size_match = re.search(r"page size of (\\d+) bytes", output)
    free_pages_match = re.search(r"^Pages free:\s+(\\d+)\\.$", output, flags=re.MULTILINE)
    speculative_pages_match = re.search(
        r"^Pages speculative:\s+(\\d+)\\.$", output, flags=re.MULTILINE
    )
    if not page_size_match or not free_pages_match or not speculative_pages_match:
        raise RuntimeError("resource governor observability unavailable: vm_stat")
    return int(page_size_match.group(1)) * (
        int(free_pages_match.group(1)) + int(speculative_pages_match.group(1))
    )


def _observe_host_resources() -> ResourceSnapshot:
    try:
        load_average_1m = os.getloadavg()[0]
    except OSError as exc:
        raise RuntimeError("resource governor observability unavailable: load average") from exc
    if not math.isfinite(load_average_1m) or load_average_1m < 0:
        raise RuntimeError("resource governor observability unavailable: load average")
    return ResourceSnapshot(
        logical_cpu_count=_positive_sysctl_u64("hw.logicalcpu"),
        load_average_1m=load_average_1m,
        available_memory_bytes=_available_memory_from_vm_stat(),
        source="macos-sysctl-vm_stat",
    )


def _require_admissible_resources(snapshot: ResourceSnapshot) -> dict[str, Any]:
    if snapshot.logical_cpu_count < 1:
        raise RuntimeError("resource governor observability unavailable: logical CPU count")
    if not math.isfinite(snapshot.load_average_1m) or snapshot.load_average_1m < 0:
        raise RuntimeError("resource governor observability unavailable: load average")
    if snapshot.available_memory_bytes < 0:
        raise RuntimeError("resource governor observability unavailable: available memory")
    if snapshot.available_memory_bytes < _MIN_AVAILABLE_MEMORY_BYTES:
        raise RuntimeError(
            "resource governor rejected available memory "
            f"({snapshot.available_memory_bytes} < {_MIN_AVAILABLE_MEMORY_BYTES})"
        )
    max_load = max(1.0, snapshot.logical_cpu_count * _MAX_LOAD_PER_LOGICAL_CPU)
    if snapshot.load_average_1m > max_load:
        raise RuntimeError(
            "resource governor rejected host load "
            f"({snapshot.load_average_1m:.2f} > {max_load:.2f})"
        )
    return {
        "observation": asdict(snapshot),
        "minimum_available_memory_bytes": _MIN_AVAILABLE_MEMORY_BYTES,
        "maximum_load_average_1m": max_load,
    }


def _fixture_command(repo_root: Path, fixture: str) -> tuple[list[str], dict[str, str]]:
    test_target, test_name = FIXTURES[fixture]
    command = [
        "cargo",
        "test",
        "--manifest-path",
        "perf-core/Cargo.toml",
        "-p",
        "metal-runtime",
        "--features",
        "metal",
        "--test",
        test_target,
        test_name,
        "--",
        "--ignored",
        "--exact",
    ]
    return command, {"CARGO_BUILD_JOBS": "1", "RUST_BACKTRACE": "0"}


def record_fixture(
    repo_root: Path,
    compile_provenance: Path,
    artifact: Path,
    manifest: Path,
    output: Path,
    fixture: str,
    timeout_seconds: int,
    resource_observer: ResourceObserver | None = None,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    artifact = artifact.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    if fixture not in FIXTURES:
        raise RuntimeError(f"unsupported fixture: {fixture}")
    if not artifact.is_file() or not manifest.is_file():
        raise RuntimeError("artifact and manifest must be regular files")
    output = _require_external_output(output, repo_root)
    head, branch = _require_clean_head(repo_root)
    provenance = _load_compile_provenance(compile_provenance, head, artifact)
    command, fixed_environment = _fixture_command(repo_root, fixture)
    resource_governor = _require_admissible_resources(
        (resource_observer or _observe_host_resources)()
    )
    environment = {**os.environ, **fixed_environment}
    if fixture == "diffusion":
        environment["METAL_RUNTIME_TEST_ARTIFACT"] = str(artifact)
        environment["METAL_RUNTIME_TEST_MANIFEST"] = str(manifest)
    else:
        environment["TERNARY_GEMM_METALLIB"] = str(artifact)
        environment["TERNARY_GEMM_MANIFEST"] = str(manifest)
    started = datetime.now(timezone.utc)
    completed = (command_runner or subprocess.run)(
        command,
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    finished = datetime.now(timezone.utc)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no test output"
        raise RuntimeError(f"bounded Metal fixture failed: {detail}")
    test_target, test_name = FIXTURES[fixture]
    record = {
        "schema_version": "pheno.metal-device-fixture.v1",
        "captured_at_start": started.isoformat().replace("+00:00", "Z"),
        "captured_at_finish": finished.isoformat().replace("+00:00", "Z"),
        "repository": "phenotype-omlx",
        "branch": branch,
        "candidate_source_head": head,
        "compile_provenance_sha256": _sha256(compile_provenance),
        "metallib_sha256": _sha256(artifact),
        "manifest_sha256": _sha256(manifest),
        "fixture": fixture,
        "test_target": test_target,
        "test_name": test_name,
        "command": command,
        "resource_governor": resource_governor,
        "device_dispatch_executed": True,
        "model_loaded": False,
        "workload_executed": False,
        "promotable": False,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--compile-provenance", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", choices=sorted(FIXTURES), required=True)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    args = parser.parse_args()
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be positive")
    try:
        record = record_fixture(
            args.repo_root,
            args.compile_provenance,
            args.artifact,
            args.manifest,
            args.output,
            args.fixture,
            args.timeout_seconds,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
