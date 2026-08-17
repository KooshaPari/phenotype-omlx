"""Pure allowlist validation shared by bounded Metal evidence edges."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_path(path: Path) -> str:
    """Return the SHA-256 of a regular artifact file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_manifest_allows_artifact(manifest: Path, artifact: Path) -> None:
    """Fail closed unless the canonical allowlist names this exact artifact."""

    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("manifest does not allow supplied artifact") from exc
    if not isinstance(document, dict) or set(document) != {"artifacts"}:
        raise RuntimeError("manifest does not allow supplied artifact")
    entries: Any = document["artifacts"]
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("manifest does not allow supplied artifact")

    expected = {"name": artifact.name, "sha256": sha256_path(artifact)}
    names: set[str] = set()
    found = False
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != set(expected):
            raise RuntimeError("manifest does not allow supplied artifact")
        if not all(isinstance(value, str) for value in entry.values()):
            raise RuntimeError("manifest does not allow supplied artifact")
        name = entry["name"]
        digest = entry["sha256"]
        if (
            Path(name).name != name
            or Path(name).suffix != ".metallib"
            or len(digest) != 64
            or not digest.isascii()
            or not all(character in "0123456789abcdefABCDEF" for character in digest)
            or name in names
        ):
            raise RuntimeError("manifest does not allow supplied artifact")
        names.add(name)
        if name == expected["name"] and digest.lower() == expected["sha256"]:
            found = True
    if not found:
        raise RuntimeError("manifest does not allow supplied artifact")
