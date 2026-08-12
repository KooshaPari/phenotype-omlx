#!/usr/bin/env python3
"""Emit the canonical allowlist manifest for compiled Metal libraries.

This is a build-edge utility only: it hashes already-compiled ``.metallib``
files and performs no model, benchmark, or device work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def build_manifest(artifact_dir: Path) -> dict[str, list[dict[str, str]]]:
    artifacts = []
    for path in sorted(artifact_dir.glob("*.metallib"), key=lambda item: item.name):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts.append({"name": path.name, "sha256": digest})
    if not artifacts:
        raise SystemExit(f"no .metallib artifacts found in {artifact_dir}")
    return {"artifacts": artifacts}


def write_manifest_once(output: Path, manifest: dict[str, list[dict[str, str]]]) -> str:
    """Atomically publish a new canonical manifest without replacing evidence."""

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"manifest output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(f"manifest output already exists: {output}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return encoded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    manifest = build_manifest(artifact_dir)
    output = args.output or artifact_dir / "metal-runtime-manifest.json"
    encoded = write_manifest_once(output, manifest)
    print(f"METAL_RUNTIME_MANIFEST={output}")
    print(f"METAL_RUNTIME_ARTIFACT_COUNT={len(manifest['artifacts'])}")
    print(hashlib.sha256(encoded.encode("utf-8")).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
