#!/usr/bin/env python3
"""Emit the canonical allowlist manifest for compiled Metal libraries.

This is a build-edge utility only: it hashes already-compiled ``.metallib``
files and performs no model, benchmark, or device work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    manifest = build_manifest(artifact_dir)
    output = args.output or artifact_dir / "metal-runtime-manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    output.write_text(encoded + "\n", encoding="utf-8")
    print(f"METAL_RUNTIME_MANIFEST={output}")
    print(f"METAL_RUNTIME_ARTIFACT_COUNT={len(manifest['artifacts'])}")
    print(hashlib.sha256(encoded.encode("utf-8")).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
