"""Contract tests for bounded Metal device-fixture evidence recording."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
RECORDER = ROOT / "scripts" / "record_metal_device_fixture.py"


def _load_recorder():
    if str(RECORDER.parent) not in sys.path:
        sys.path.insert(0, str(RECORDER.parent))
    spec = importlib.util.spec_from_file_location(
        "record_metal_device_fixture", RECORDER
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git_repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Fixture Test"], check=True
    )
    (path / "tracked.txt").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)


def _valid_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    _git_repository(repo)
    fixture_source = (
        repo / "perf-core" / "metal-runtime" / "tests" / "diffusion_dispatch.rs"
    )
    fixture_source.parent.mkdir(parents=True)
    fixture_source.write_text(
        '#[ignore = "fixture"]\n'
        "fn diffusion_three_stage_fixture_matches_oracle() {}\n"
        "// METAL_RUNTIME_TEST_ARTIFACT METAL_RUNTIME_TEST_MANIFEST\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(repo), "add", str(fixture_source)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "fixture contract"], check=True
    )
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    artifact = tmp_path / "metal-runtime.metallib"
    artifact.write_bytes(b"fixture")
    manifest = tmp_path / "metal-runtime-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "name": artifact.name,
                        "sha256": __import__("hashlib")
                        .sha256(artifact.read_bytes())
                        .hexdigest(),
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "candidate_source_head": head,
                "build_checkout_head": head,
                "status": "current_head_compile_only",
                "metallib_sha256": __import__("hashlib")
                .sha256(artifact.read_bytes())
                .hexdigest(),
                "workload_executed": False,
                "device_dispatch_executed": False,
                "model_loaded": False,
            }
        ),
        encoding="utf-8",
    )
    return repo, artifact, manifest, provenance, tmp_path / "fixture-evidence.json"


def test_resource_governor_rejects_unavailable_observability_before_cargo(
    tmp_path: Path,
) -> None:
    recorder = _load_recorder()
    repo, artifact, manifest, provenance, output = _valid_inputs(tmp_path)
    cargo_calls: list[object] = []

    def unavailable_observer():
        raise RuntimeError("resource governor observability unavailable: vm_stat")

    def cargo_runner(*args, **kwargs):
        cargo_calls.append((args, kwargs))
        raise AssertionError("cargo must not run when observability is unavailable")

    with pytest.raises(
        RuntimeError, match="resource governor observability unavailable"
    ):
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


def test_resource_governor_rejects_constrained_host_before_cargo(
    tmp_path: Path,
) -> None:
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
        raise AssertionError(
            "cargo must not run when host load exceeds the governor limit"
        )

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


def test_vm_stat_observer_counts_only_reclaimable_pages() -> None:
    recorder = _load_recorder()
    macos_vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                     7769.
Pages active:                                 277190.
Pages inactive:                               263191.
Pages speculative:                               867.
Pages purgeable:                               253144.
"""

    available_memory_bytes = recorder._available_memory_from_vm_stat(macos_vm_stat)

    assert available_memory_bytes == (7769 + 867 + 253144) * 16384
    assert available_memory_bytes < 4 * 1024**3


def test_preflight_emits_admission_evidence_without_running_cargo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder = _load_recorder()
    repo, _, _, _, output = _valid_inputs(tmp_path)

    def admissible_observer():
        return recorder.ResourceSnapshot(
            logical_cpu_count=8,
            load_average_1m=1.0,
            available_memory_bytes=8 * 1024**3,
            source="test",
        )

    original_run = recorder.subprocess.run

    def no_cargo_runner(command, *args, **kwargs):
        if command[0] == "cargo":
            raise AssertionError("preflight must never invoke cargo")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(recorder, "_observe_host_resources", admissible_observer)
    monkeypatch.setattr(recorder.subprocess, "run", no_cargo_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RECORDER),
            "--preflight",
            "--repo-root",
            str(repo),
            "--output",
            str(output),
            "--fixture",
            "diffusion",
        ],
    )

    assert recorder.main() == 0
    record = json.loads(capsys.readouterr().out)
    assert record["schema_version"] == "pheno.metal-device-fixture-preflight.v1"
    assert record["device_dispatch_executed"] is False
    assert record["model_loaded"] is False
    assert record["workload_executed"] is False
    assert record["promotable"] is False
    assert "command" not in record
    assert "metallib_sha256" not in record
    assert "manifest_sha256" not in record
    assert json.loads(output.read_text(encoding="utf-8")) == record


def test_preflight_requires_a_clean_named_head(tmp_path: Path) -> None:
    recorder = _load_recorder()
    repo, _, _, _, output = _valid_inputs(tmp_path)
    (repo / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="repository must be clean"):
        recorder.preflight_fixture(
            repo,
            output,
            "diffusion",
            resource_observer=lambda: recorder.ResourceSnapshot(
                logical_cpu_count=8,
                load_average_1m=1.0,
                available_memory_bytes=8 * 1024**3,
                source="test",
            ),
        )
    assert not output.exists()


def test_preflight_records_resource_denial_without_dispatching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _load_recorder()
    repo, _, _, _, output = _valid_inputs(tmp_path)
    original_run = recorder.subprocess.run

    def no_cargo_runner(command, *args, **kwargs):
        if command[0] == "cargo":
            raise AssertionError("resource-denial preflight must never invoke cargo")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(recorder.subprocess, "run", no_cargo_runner)
    with pytest.raises(RuntimeError, match="resource governor rejected host load"):
        recorder.preflight_fixture(
            repo,
            output,
            "diffusion",
            resource_observer=lambda: recorder.ResourceSnapshot(
                logical_cpu_count=8,
                load_average_1m=8.0,
                available_memory_bytes=8 * 1024**3,
                source="test",
            ),
        )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["admitted"] is False
    assert (
        record["rejection_reason"]
        == "resource governor rejected host load (8.00 > 6.00)"
    )
    assert record["resource_governor"]["observation"]["load_average_1m"] == 8.0
    assert record["device_dispatch_executed"] is False
    assert record["model_loaded"] is False
    assert record["workload_executed"] is False
    assert record["promotable"] is False


def test_preflight_does_not_write_evidence_for_an_invalid_fixture(
    tmp_path: Path,
) -> None:
    recorder = _load_recorder()
    repo, _, _, _, output = _valid_inputs(tmp_path)

    with pytest.raises(RuntimeError, match="unsupported fixture"):
        recorder.preflight_fixture(repo, output, "invalid-fixture")
    assert not output.exists()


def test_preflight_records_missing_fixture_contract_without_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _load_recorder()
    repo, _, _, _, output = _valid_inputs(tmp_path)
    fixture_source = (
        repo / "perf-core" / "metal-runtime" / "tests" / "diffusion_dispatch.rs"
    )
    fixture_source.write_text("// wrong fixture source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", str(fixture_source)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "fixture source"], check=True
    )

    with pytest.raises(RuntimeError, match="fixture source contract unavailable"):
        recorder.preflight_fixture(
            repo,
            output,
            "diffusion",
            resource_observer=lambda: recorder.ResourceSnapshot(
                logical_cpu_count=8,
                load_average_1m=1.0,
                available_memory_bytes=8 * 1024**3,
                source="test",
            ),
        )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["admitted"] is False
    assert "fixture source contract unavailable" in record["rejection_reason"]
    assert record["device_dispatch_executed"] is False
    assert record["promotable"] is False


def test_record_fixture_refuses_a_race_created_output(tmp_path: Path) -> None:
    recorder = _load_recorder()
    repo, artifact, manifest, provenance, output = _valid_inputs(tmp_path)

    def race_runner(*args, **kwargs):
        output.write_text("preserve-raced-evidence", encoding="utf-8")
        return subprocess.CompletedProcess(
            args[0], 0, stdout="fixture passed", stderr=""
        )

    with pytest.raises(RuntimeError, match="output already exists"):
        recorder.record_fixture(
            repo,
            provenance,
            artifact,
            manifest,
            output,
            "diffusion",
            90,
            resource_observer=lambda: recorder.ResourceSnapshot(
                logical_cpu_count=8,
                load_average_1m=1.0,
                available_memory_bytes=8 * 1024**3,
                source="test",
            ),
            command_runner=race_runner,
        )

    assert output.read_text(encoding="utf-8") == "preserve-raced-evidence"


def test_record_fixture_rejects_unlisted_artifact_before_cargo(tmp_path: Path) -> None:
    recorder = _load_recorder()
    repo, artifact, manifest, provenance, output = _valid_inputs(tmp_path)
    manifest.write_text('{"artifacts":[]}\n', encoding="utf-8")
    cargo_calls: list[object] = []

    def cargo_runner(*args, **kwargs):
        cargo_calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args[0], 0, stdout="fixture passed", stderr=""
        )

    with pytest.raises(RuntimeError, match="manifest does not allow supplied artifact"):
        recorder.record_fixture(
            repo,
            provenance,
            artifact,
            manifest,
            output,
            "diffusion",
            90,
            resource_observer=lambda: recorder.ResourceSnapshot(
                logical_cpu_count=8,
                load_average_1m=1.0,
                available_memory_bytes=8 * 1024**3,
                source="test",
            ),
            command_runner=cargo_runner,
        )

    assert cargo_calls == []


def test_record_fixture_rejects_a_malformed_unrelated_manifest_entry(
    tmp_path: Path,
) -> None:
    recorder = _load_recorder()
    repo, artifact, manifest, provenance, output = _valid_inputs(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["artifacts"].append({"name": "../escape.metallib", "sha256": "0" * 64})
    manifest.write_text(json.dumps(document) + "\n", encoding="utf-8")
    cargo_calls: list[object] = []

    def cargo_runner(*args, **kwargs):
        cargo_calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args[0], 0, stdout="fixture passed", stderr=""
        )

    with pytest.raises(RuntimeError, match="manifest does not allow supplied artifact"):
        recorder.record_fixture(
            repo,
            provenance,
            artifact,
            manifest,
            output,
            "diffusion",
            90,
            resource_observer=lambda: recorder.ResourceSnapshot(
                logical_cpu_count=8,
                load_average_1m=1.0,
                available_memory_bytes=8 * 1024**3,
                source="test",
            ),
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
