"""Contract tests for bounded Metal device-fixture evidence recording."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[2]
RECORDER = ROOT / "scripts" / "record_metal_device_fixture.py"


def _load_recorder():
    spec = importlib.util.spec_from_file_location("record_metal_device_fixture", RECORDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git_repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Fixture Test"], check=True)
    (path / "tracked.txt").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)


def _valid_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    _git_repository(repo)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    artifact = tmp_path / "metal-runtime.metallib"
    artifact.write_bytes(b"fixture")
    manifest = tmp_path / "metal-runtime-manifest.json"
    manifest.write_text('{"artifacts":[]}\n', encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "candidate_source_head": head,
                "build_checkout_head": head,
                "status": "current_head_compile_only",
                "metallib_sha256": __import__("hashlib").sha256(artifact.read_bytes()).hexdigest(),
                "workload_executed": False,
                "device_dispatch_executed": False,
                "model_loaded": False,
            }
        ),
        encoding="utf-8",
    )
    return repo, artifact, manifest, provenance, tmp_path / "fixture-evidence.json"


def test_resource_governor_rejects_unavailable_observability_before_cargo(tmp_path: Path) -> None:
    recorder = _load_recorder()
    repo, artifact, manifest, provenance, output = _valid_inputs(tmp_path)
    cargo_calls: list[object] = []

    def unavailable_observer():
        raise RuntimeError("resource governor observability unavailable: vm_stat")

    def cargo_runner(*args, **kwargs):
        cargo_calls.append((args, kwargs))
        raise AssertionError("cargo must not run when observability is unavailable")

    with pytest.raises(RuntimeError, match="resource governor observability unavailable"):
        recorder.record_fixture(
            repo,
            provenance,
            artifact,
            manifest,
            output,
            "diffusion",
            90,
            resource_observer=unavailable_observer,
            command_runner=cargo_runner,
        )
    assert cargo_calls == []


def test_resource_governor_rejects_constrained_host_before_cargo(tmp_path: Path) -> None:
    recorder = _load_recorder()
    repo, artifact, manifest, provenance, output = _valid_inputs(tmp_path)
    cargo_calls: list[object] = []

    def constrained_observer():
        return recorder.ResourceSnapshot(
            logical_cpu_count=8,
            load_average_1m=8.0,
            available_memory_bytes=8 * 1024**3,
            source="test",
        )

    def cargo_runner(*args, **kwargs):
        cargo_calls.append((args, kwargs))
        raise AssertionError("cargo must not run when host load exceeds the governor limit")

    with pytest.raises(RuntimeError, match="resource governor rejected host load"):
        recorder.record_fixture(
            repo,
            provenance,
            artifact,
            manifest,
            output,
            "diffusion",
            90,
            resource_observer=constrained_observer,
            command_runner=cargo_runner,
        )
    assert cargo_calls == []


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
