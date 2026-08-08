"""Contract tests for bounded Metal device-fixture evidence recording."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[2]
RECORDER = ROOT / "scripts" / "record_metal_device_fixture.py"


def _git_repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Fixture Test"], check=True)
    (path / "tracked.txt").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)


def test_rejects_compile_provenance_from_a_different_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git_repository(repo)
    artifact = tmp_path / "metal-runtime.metallib"
    artifact.write_bytes(b"fixture")
    manifest = tmp_path / "metal-runtime-manifest.json"
    manifest.write_text('{"artifacts":[]}\n', encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "candidate_source_head": "0" * 40,
                "build_checkout_head": "0" * 40,
                "status": "current_head_compile_only",
                "metallib_sha256": "0" * 64,
                "workload_executed": False,
                "device_dispatch_executed": False,
                "model_loaded": False,
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(RECORDER),
            "--repo-root",
            str(repo),
            "--compile-provenance",
            str(provenance),
            "--artifact",
            str(artifact),
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "fixture-evidence.json"),
            "--fixture",
            "diffusion",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "compile provenance is not bound to current HEAD" in result.stderr
