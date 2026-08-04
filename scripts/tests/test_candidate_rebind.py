"""Contract tests for immutable current-head candidate rebind preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import scripts.prepare_candidate_rebind as rebind
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


def _branch_for(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "branch", "--show-current"],
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


def _evidence(head: str, repo: Path) -> dict:
    return _with_integrity(
        {
            "schema_version": "0.1",
            "evidence_label": "live_verified",
            "model": "mlx-community/Qwen3.5-0.8B-OptiQ-4bit",
            "candidate": {
                "repository": repo.name,
                "source_head": head,
                "branch": _branch_for(repo),
            },
            "harbor": {
                "job_id": "fixture-job",
                "trial_name": "omlx-niah-api-smoke__fixture",
                "task": "omlx/niah-api-smoke",
                "environment": "apple-container",
                "n_trials": 1,
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
            "authorization": {
                "window_id": "window-fixture-123",
                "sidecar_path": "authorization.json",
                "sidecar_sha256": hashlib.sha256(
                    (repo / "authorization.json").read_bytes()
                ).hexdigest(),
            },
            "artifacts": [
                {
                    "path": "result.json",
                    "sha256": hashlib.sha256((repo / "result.json").read_bytes()).hexdigest(),
                }
            ],
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
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Candidate Rebind Test"],
        check=True,
    )
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    (path / "result.json").write_text('{"fixture":true}\n', encoding="utf-8")
    (path / "authorization.json").write_text(
        '{"window_id":"window-fixture-123","approved":true}\n', encoding="utf-8"
    )
    config = path / "config"
    config.mkdir()
    (config / "smoke_models.json").write_text(
        json.dumps(
            {
                "defaults": {"mlx_hf": "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"},
                "roles": {"readiness": "mlx_hf"},
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "add",
            "tracked.txt",
            "result.json",
            "authorization.json",
            "config",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    return path


def test_prepares_review_only_record_from_current_head_evidence(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    head = _head_for(repo)
    evidence_path = _write_json(tmp_path / "evidence.json", _evidence(head, repo))
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


def test_records_validated_evidence_bytes_when_input_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _git_repository(tmp_path / "repo")
    head = _head_for(repo)
    evidence_path = _write_json(tmp_path / "evidence.json", _evidence(head, repo))
    metal_path = _write_json(tmp_path / "metal.json", _metal(head))
    validated_digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    original_validate_metal = rebind._validate_metal

    def replace_evidence_after_validation(document: dict, current_head: str) -> dict:
        evidence_path.write_text('{"replacement":true}\n', encoding="utf-8")
        return original_validate_metal(document, current_head)

    monkeypatch.setattr(rebind, "_validate_metal", replace_evidence_after_validation)

    record = prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)

    assert record["evidence"]["file_sha256"] == validated_digest
    assert record["evidence"]["input"]["sha256"] == validated_digest
    assert record["evidence"]["authorization_sidecar"]
    assert record["evidence"]["artifacts"]


def test_rejects_retried_harbor_evidence(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence = _evidence(_head_for(repo), repo)
    evidence["harbor"]["n_retries"] = 1
    evidence = _with_integrity(evidence)
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match="retries"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_rejects_boolean_harbor_reward(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence = _evidence(_head_for(repo), repo)
    evidence["harbor"]["reward"] = True
    evidence = _with_integrity(evidence)
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match="Harbor reward"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_rejects_non_current_head_evidence(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence_path = _write_json(tmp_path / "evidence.json", _evidence("a" * 40, repo))
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
    evidence_path = _write_json(tmp_path / "evidence.json", _evidence(head, repo))
    metal_path = _write_json(tmp_path / "metal.json", _metal(head))
    (repo / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
    output_path = tmp_path / "record.json"

    with pytest.raises(CandidateRebindError, match="clean working tree"):
        prepare_rebind(evidence_path, metal_path, output_path, repo)

    assert not output_path.exists()


def test_rejects_qwen25_evidence_even_with_valid_digest(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence = _evidence(_head_for(repo), repo)
    evidence["model"] = "mlx-community/Qwen2.5-0.5B-4bit"
    evidence = _with_integrity(evidence)
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match="Qwen3.5"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_rejects_noncanonical_qwen35_model(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence = _evidence(_head_for(repo), repo)
    evidence["model"] = "evil/Qwen3.5-test"
    evidence = _with_integrity(evidence)
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match="canonical Qwen3.5"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_rejects_non_niah_or_multi_trial_evidence(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence = _evidence(_head_for(repo), repo)
    evidence["harbor"]["task"] = "omlx/other"
    evidence["harbor"]["n_trials"] = 2
    evidence = _with_integrity(evidence)
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match="Harbor task"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_rejects_multi_trial_evidence(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence = _evidence(_head_for(repo), repo)
    evidence["harbor"]["n_trials"] = 2
    evidence_path = _write_json(tmp_path / "evidence.json", _with_integrity(evidence))
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match="Harbor n_trials"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("repository", "other-repository", "candidate repository"),
        ("branch", "other-branch", "candidate branch"),
    ],
)
def test_rejects_mismatched_candidate_identity(
    tmp_path: Path, field: str, replacement: str, error: str
) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence = _evidence(_head_for(repo), repo)
    evidence["candidate"][field] = replacement
    evidence_path = _write_json(tmp_path / "evidence.json", _with_integrity(evidence))
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match=error):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_rejects_missing_harbor_artifact(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence = _evidence(_head_for(repo), repo)
    evidence["artifacts"][0]["path"] = "missing-result.json"
    evidence = _with_integrity(evidence)
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match="regular file"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_rejects_integer_for_boolean_harbor_gate(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence = _evidence(_head_for(repo), repo)
    evidence["harbor"]["fallback_applied"] = 0
    evidence = _with_integrity(evidence)
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match="fallback_applied"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_rejects_unbound_authorization_sidecar(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence = _evidence(_head_for(repo), repo)
    evidence["authorization"]["sidecar_path"] = "missing-authorization.json"
    evidence = _with_integrity(evidence)
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match="authorization sidecar"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_rejects_mismatched_authorization_sidecar_window(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    sidecar = repo / "authorization.json"
    sidecar.write_text(
        '{"window_id":"other-window"}\n', encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(repo), "add", "authorization.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "mismatched sidecar"], check=True)
    head = _head_for(repo)
    evidence_path = _write_json(tmp_path / "evidence.json", _evidence(head, repo))
    metal_path = _write_json(tmp_path / "metal.json", _metal(head))

    with pytest.raises(CandidateRebindError, match="authorization sidecar window_id"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_rejects_unapproved_authorization_sidecar(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    sidecar = repo / "authorization.json"
    sidecar.write_text(
        '{"window_id":"window-fixture-123","approved":false}\n', encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(repo), "add", "authorization.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "unapproved sidecar"], check=True)
    head = _head_for(repo)
    evidence_path = _write_json(tmp_path / "evidence.json", _evidence(head, repo))
    metal_path = _write_json(tmp_path / "metal.json", _metal(head))

    with pytest.raises(CandidateRebindError, match="authorization sidecar approved"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_rejects_in_repo_symlink_artifact(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    link = repo / "result-link.json"
    link.symlink_to(repo / "result.json")
    subprocess.run(["git", "-C", str(repo), "add", link.name], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "tracked-symlink"], check=True)
    evidence = _evidence(_head_for(repo), repo)
    evidence["artifacts"][0] = {
        "path": link.name,
        "sha256": hashlib.sha256((repo / "result.json").read_bytes()).hexdigest(),
    }
    evidence = _with_integrity(evidence)
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match="symlink"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_rejects_embedded_nul_artifact_path(tmp_path: Path) -> None:
    repo = _git_repository(tmp_path / "repo")
    evidence = _evidence(_head_for(repo), repo)
    evidence["artifacts"][0]["path"] = "result\x00.json"
    evidence = _with_integrity(evidence)
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)
    metal_path = _write_json(tmp_path / "metal.json", _metal(_head_for(repo)))

    with pytest.raises(CandidateRebindError, match="path must not contain NUL"):
        prepare_rebind(evidence_path, metal_path, tmp_path / "record.json", repo)


def test_refuses_historical_manifest_and_existing_output(tmp_path: Path) -> None:
    fixture_repo = _git_repository(tmp_path / "fixture-repo")
    head = _head()
    evidence_path = _write_json(tmp_path / "evidence.json", _evidence(head, fixture_repo))
    metal_path = _write_json(tmp_path / "metal.json", _metal(head))
    historical = ROOT / "docs/sessions/20260718-metal-model-runtime/candidate-manifest.json"

    with pytest.raises(CandidateRebindError, match="historical candidate manifest"):
        prepare_rebind(evidence_path, metal_path, historical, ROOT)

    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(CandidateRebindError, match="must not already exist"):
        prepare_rebind(evidence_path, metal_path, existing, ROOT)
