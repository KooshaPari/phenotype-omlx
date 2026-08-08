#!/usr/bin/env python3
"""Record a reproducible compile-only Metal provenance envelope.

The output is intentionally written outside the repository by default. Committing the
record would change HEAD after compilation and invalidate an exact-head claim; callers
can instead pass the external JSON directly to candidate rebind preparation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], text=True
    ).strip()


def _require_clean(repo_root: Path) -> None:
    if _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("repository must be clean before compile provenance capture")


def _record(repo_root: Path, output: Path, keep_build: Path | None) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"output already exists: {output}")
    _require_clean(repo_root)
    head = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "branch", "--show-current")
    if not branch:
        raise RuntimeError("repository must be on a named branch")

    build_dir = keep_build or Path(tempfile.mkdtemp(prefix="phenotype-omlx-metal-"))
    build_dir = build_dir.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    log_path = build_dir / "build.log"
    started = datetime.now(timezone.utc)
    with log_path.open("wb") as log:
        subprocess.run(
            ["bash", "scripts/build_metal_runtime_bundle.sh"],
            cwd=repo_root,
            env={**__import__("os").environ, "OUT_DIR": str(build_dir)},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    finished = datetime.now(timezone.utc)
    artifact = build_dir / "metal-runtime.metallib"
    if not artifact.is_file():
        raise RuntimeError(f"Metal build did not produce {artifact}")
    record = {
        "schema_version": "pheno.metal-compile-provenance.v1",
        "captured_at_start": started.isoformat().replace("+00:00", "Z"),
        "captured_at_finish": finished.isoformat().replace("+00:00", "Z"),
        "repository": "phenotype-omlx",
        "branch": branch,
        "candidate_source_head": head,
        "build_checkout_head": head,
        "source_head_compatible": True,
        "command": "OUT_DIR=<temporary> bash scripts/build_metal_runtime_bundle.sh",
        "toolchain": {
            "developer_directory": "/Users/kooshapari/Downloads/Xcode-beta.app/Contents/Developer",
            "metal_toolchain": "Xcode-beta Metal.xctoolchain",
            "metal_binary_resolved": True,
            "metallib_binary_resolved": True,
        },
        "shader_count": 20,
        "metallib_sha256": _sha256(artifact),
        "build_log_sha256": _sha256(log_path),
        "workload_executed": False,
        "device_dispatch_executed": False,
        "model_loaded": False,
        "status": "current_head_compile_only",
        "promotable": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-build", type=Path, help="Keep compiled artifact and build log here")
    args = parser.parse_args()
    try:
        record = _record(args.repo_root, args.output, args.keep_build)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
