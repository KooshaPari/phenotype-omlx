"""Contracts for immutable Metal allowlist-manifest publication."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "manifest_metal_runtime_artifacts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("metal_manifest", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_publication_refuses_existing_or_symlink_output(
    tmp_path: Path,
) -> None:
    module = _load_module()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "fixture.metallib").write_bytes(b"fixture")
    protected = tmp_path / "protected.json"
    protected.write_text("preserve-me", encoding="utf-8")
    output = tmp_path / "manifest.json"
    output.symlink_to(protected)

    with pytest.raises(FileExistsError, match="manifest output already exists"):
        module.write_manifest_once(output, module.build_manifest(artifacts))

    assert protected.read_text(encoding="utf-8") == "preserve-me"
