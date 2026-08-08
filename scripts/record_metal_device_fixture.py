#!/usr/bin/env python3
"""Run one bounded Metal fixture and record external, non-promotable evidence.

The recorder requires a current-head compile-only provenance record, an
allowlisted metallib, and one of the small ignored parity fixtures.  It never
loads a model, starts Harbor, or runs an evaluation workload.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


FIXTURES = {
    "diffusion": ("diffusion_dispatch", "diffusion_three_stage_fixture_matches_oracle"),
    "ternary-small": ("ternary", "metal_matches_scalar_reference"),
    "ternary-edge": ("ternary", "metal_matches_edge_shape_with_all_packed_codes"),
}


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
    environment = {**os.environ, **fixed_environment}
    if fixture == "diffusion":
        environment["METAL_RUNTIME_TEST_ARTIFACT"] = str(artifact)
        environment["METAL_RUNTIME_TEST_MANIFEST"] = str(manifest)
    else:
        environment["TERNARY_GEMM_METALLIB"] = str(artifact)
        environment["TERNARY_GEMM_MANIFEST"] = str(manifest)
    started = datetime.now(timezone.utc)
    completed = subprocess.run(
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
