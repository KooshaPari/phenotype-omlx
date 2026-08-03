"""Contract tests for immutable current-head candidate rebind preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.prepare_candidate_rebind import CandidateRebindError, prepare_rebind


ROOT = Path(__file__).parents[2]


def _head() -> str:
    return _head_for(ROOT)


def _head_for(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _with_integrity(document: dict) -> dict:
    payload = {key: value for key, value in document.items() if key != "integrity"}
    document["integrity"] = {
        "canonical_sha256": hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    }
    return document


def _evidence(head: str) -> dict:
    return _with_integrity(
        {
            "schema_version": "0.1",
            "evidence_label": "live_verified",
            "model": "mlx-community/Qwen3.5-0.8B-OptiQ-4bit",
            "candidate": {"source_head": head, "branch": "fixture/current-head"},
            "harbor": {
                "job_id": "fixture-job",
                "trial_name": "omlx-niah-api-smoke__fixture",
                "environment": "apple-container",
                "n_errors": 0,
                "n_retries": 0,
                "fallback_applied": False,
                "reward": 1.0,
                "pass_at_1": 1.0,
                "requested_context_tokens": 8192,
                "prompt_tokens": 8192,
                "context_tokens_exact": True,
                "prompt_sha256": "d" * 64,
            },
            "artifacts": [{"path": "result.json", "sha256": "a" * 64}],
        }
    )


def _metal(head: str) -> dict:
    return {
        "schema_version": "pheno.metal-compile-provenance.v1",
        "candidate_source_head": head,
        "build_checkout_head": head,
        "source_head_compatible": True,
        "shader_count": 20,
        "metallib_sha256": "b" * 64,
        "build_log_sha256": "c" * 64,
        "workload_executed": False,
        "device_dispatch_executed": False,
        "model_loaded": False,
        "status": "current_head_compile_only",
        "promotable": False,
    }


def _write_json(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _git_repository(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Candidate Rebind Test"], check=True)
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    return path


def test_prepares_review_only_record_from_current_head_evidence(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    head = _head_for(repo)
    evidence_path = _write_json(tmp_path / "evidence.json", _evidence(head))
    metal_path = _write_json(tmp_path / "metal.json", _metal(head))
    output_path = tmp_path / "candidate-rebind.json"

    record = prepare_rebind(evidence_path, metal_path, output_path, repo)

    assert output_path.is_file()
    assert record["schema_version"] == "pheno.candidate-rebind-review.v1"
    assert record["candidate"]["head"] == head
    assert record["evidence"]["model"] == "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
    assert record["promotion"]["verdict"] == "review_required"
    assert record["promotion"]["promotable"] is False
    assert record["integrity"]["canonical_sha256"]


def test_rejects_retried_harbor_evidence(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence = _evidence(_head_for(repo))
    evidence["harbor"]["n_retries"] = 1
    evidence = _with_integrity(evidence)
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match="retries"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_rejects_boolean_harbor_reward(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence = _evidence(_head_for(repo))
    evidence["harbor"]["reward"] = True
    evidence = _with_integrity(evidence)
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match="Harbor reward"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_rejects_non_current_head_evidence(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence_path = _write_json(tmp_path / "evidence.json", _evidence("a" * 40))
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match="current repository HEAD"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_rejects_dirty_checkout_before_writing_record(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    evidence_path = _write_json(tmp_path / "evidence.json", _evidence(head))
    metal_path = _write_json(tmp_path / "metal.json", _metal(head))
    (repo / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
    output_path = tmp_path / "record.json"

    with pytest.raises(CandidateRebindError, match="clean working tree"):
        prepare_rebind(evidence_path, metal_path, output_path, repo)

    assert not output_path.exists()


def test_rejects_qwen25_evidence_even_with_valid_digest(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence = _evidence(_head_for(repo))
    evidence["model"] = "mlx-community/Qwen2.5-0.5B-4bit"
    evidence = _with_integrity(evidence)
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match="Qwen3.5"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_refuses_historical_manifest_and_existing_output(tmp_path: Path) -> None:
    head = _head()
    evidence_path = _write_json(tmp_path / "evidence.json", _evidence(head))
    metal_path = _write_json(tmp_path / "metal.json", _metal(head))
    historical = ROOT / "docs/sessions/20260718-metal-model-runtime/candidate-manifest.json"

    with pytest.raises(CandidateRebindError, match="historical candidate manifest"):
        prepare_rebind(evidence_path, metal_path, historical, ROOT)

    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(CandidateRebindError, match="must not already exist"):
        prepare_rebind(evidence_path, metal_path, existing, ROOT)
