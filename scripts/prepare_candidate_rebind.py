#!/usr/bin/env python3
"""Prepare immutable current-head evidence for an independent promotion review.

The resulting record is non-promotable and never changes historical manifests or runs work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from scripts.rebind_inputs import CandidateRebindError, InputSnapshot, canonical_digest, is_sha256
from scripts.rebind_inputs import (
    load_json_snapshot,
    load_repository_json_snapshot,
    repository_file_descriptor,
)


SCHEMA_VERSION = "pheno.candidate-rebind-review.v1"
_COMMIT_LENGTH = 40


def _canonical_digest(document: Mapping[str, Any]) -> str:
    return canonical_digest(document)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateRebindError(f"{field} must be an object")
    return value


def _current_head(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CandidateRebindError("cannot read current repository HEAD") from exc
    head = completed.stdout.strip()
    if len(head) != _COMMIT_LENGTH or any(ch not in "0123456789abcdef" for ch in head):
        raise CandidateRebindError("current repository HEAD is not a full commit SHA")
    return head


def _current_branch(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CandidateRebindError("cannot read current repository branch") from exc
    branch = completed.stdout.strip()
    if not branch:
        raise CandidateRebindError("candidate preparation requires a named branch")
    return branch


def _require_clean_worktree(repo_root: Path) -> None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CandidateRebindError("cannot inspect repository working tree") from exc
    if completed.stdout:
        raise CandidateRebindError("candidate preparation requires a clean working tree")


def _require_exact(value: Any, expected: Any, field: str) -> None:
    if value != expected:
        raise CandidateRebindError(f"{field} must equal {expected!r}")


def _require_integer(value: Any, expected: int, field: str) -> None:
    if type(value) is not int or value != expected:
        raise CandidateRebindError(f"{field} must equal integer {expected}")


def _require_score(value: Any, expected: float, field: str) -> None:
    if type(value) not in (int, float) or value != expected:
        raise CandidateRebindError(f"{field} must equal numeric {expected}")


def _require_boolean(value: Any, expected: bool, field: str) -> None:
    if type(value) is not bool or value is not expected:
        raise CandidateRebindError(f"{field} must equal boolean {expected}")


def _canonical_readiness_model(repo_root: Path) -> str:
    config = load_json_snapshot(
        repo_root / "config" / "smoke_models.json", "smoke model SSOT"
    ).document
    defaults = _mapping(config.get("defaults"), "smoke model defaults")
    roles = _mapping(config.get("roles"), "smoke model roles")
    default_key = roles.get("readiness")
    if not isinstance(default_key, str) or not default_key:
        raise CandidateRebindError("smoke model readiness role is missing")
    model = defaults.get(default_key)
    if not isinstance(model, str) or not model:
        raise CandidateRebindError("smoke model readiness default is missing")
    return model


def _validate_evidence(
    document: Mapping[str, Any], current_head: str, current_branch: str, repo_root: Path
) -> dict[str, Any]:
    _require_exact(document.get("schema_version"), "0.1", "evidence schema_version")
    _require_exact(document.get("evidence_label"), "live_verified", "evidence_label")
    declared_digest = _mapping(document.get("integrity"), "evidence integrity").get(
        "canonical_sha256"
    )
    if not is_sha256(declared_digest) or declared_digest != _canonical_digest(document):
        raise CandidateRebindError("evidence canonical SHA-256 does not match")

    model = document.get("model")
    canonical_model = _canonical_readiness_model(repo_root)
    if model != canonical_model:
        raise CandidateRebindError(
            "evidence model must equal the canonical Qwen3.5 readiness model"
        )

    candidate = _mapping(document.get("candidate"), "evidence candidate")
    _require_exact(candidate.get("repository"), repo_root.name, "evidence candidate repository")
    _require_exact(candidate.get("branch"), current_branch, "evidence candidate branch")
    _require_exact(
        candidate.get("source_head"),
        current_head,
        "evidence source head vs current repository HEAD",
    )
    harbor = _mapping(document.get("harbor"), "Harbor evidence")
    for field in ("job_id", "trial_name", "prompt_sha256"):
        if not isinstance(harbor.get(field), str) or not harbor[field]:
            raise CandidateRebindError(f"Harbor {field} must be non-empty")
    if not is_sha256(harbor["prompt_sha256"]):
        raise CandidateRebindError("Harbor prompt_sha256 must be a lowercase SHA-256")
    _require_exact(harbor.get("environment"), "apple-container", "Harbor environment")
    _require_exact(harbor.get("task"), "omlx/niah-api-smoke", "Harbor task")
    _require_integer(harbor.get("n_trials"), 1, "Harbor n_trials")
    _require_integer(harbor.get("n_errors"), 0, "Harbor n_errors")
    _require_integer(harbor.get("n_retries"), 0, "Harbor retries")
    _require_boolean(harbor.get("fallback_applied"), False, "Harbor fallback_applied")
    _require_score(harbor.get("reward"), 1.0, "Harbor reward")
    _require_score(harbor.get("pass_at_1"), 1.0, "Harbor pass_at_1")
    _require_integer(
        harbor.get("requested_context_tokens"), 8192, "Harbor requested_context_tokens"
    )
    _require_integer(harbor.get("prompt_tokens"), 8192, "Harbor prompt_tokens")
    _require_boolean(harbor.get("context_tokens_exact"), True, "Harbor context_tokens_exact")

    authorization = _mapping(document.get("authorization"), "evidence authorization")
    if not isinstance(authorization.get("window_id"), str) or not authorization["window_id"]:
        raise CandidateRebindError("evidence authorization window_id must be non-empty")
    sidecar_snapshot = load_repository_json_snapshot(
        repo_root,
        {
            "path": authorization.get("sidecar_path"),
            "sha256": authorization.get("sidecar_sha256"),
        },
        "evidence authorization sidecar",
    )
    sidecar = {
        "path": authorization["sidecar_path"],
        "sha256": sidecar_snapshot.sha256,
    }
    _require_exact(
        sidecar_snapshot.document.get("window_id"),
        authorization["window_id"],
        "evidence authorization sidecar window_id",
    )

    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CandidateRebindError("evidence artifacts must be a non-empty array")
    artifact_descriptors: list[dict[str, str]] = []
    for artifact in artifacts:
        item = _mapping(artifact, "evidence artifact")
        artifact_descriptors.append(
            repository_file_descriptor(repo_root, item, "evidence artifact")
        )

    return {
        "model": model,
        "harbor": dict(harbor),
        "authorization": dict(authorization),
        "authorization_sidecar": sidecar,
        "artifacts": artifact_descriptors,
    }


def _validate_metal(document: Mapping[str, Any], current_head: str) -> dict[str, Any]:
    _require_exact(
        document.get("schema_version"),
        "pheno.metal-compile-provenance.v1",
        "Metal provenance schema_version",
    )
    _require_exact(
        document.get("candidate_source_head"),
        current_head,
        "Metal candidate_source_head vs current repository HEAD",
    )
    _require_exact(
        document.get("build_checkout_head"),
        current_head,
        "Metal build_checkout_head vs current repository HEAD",
    )
    _require_boolean(document.get("source_head_compatible"), True, "Metal source_head_compatible")
    _require_exact(document.get("shader_count"), 20, "Metal shader_count")
    for field in ("metallib_sha256", "build_log_sha256"):
        if not is_sha256(document.get(field)):
            raise CandidateRebindError(f"Metal {field} must be a lowercase SHA-256")
    _require_boolean(document.get("workload_executed"), False, "Metal workload_executed")
    _require_boolean(
        document.get("device_dispatch_executed"), False, "Metal device_dispatch_executed"
    )
    _require_boolean(document.get("model_loaded"), False, "Metal model_loaded")
    _require_boolean(document.get("promotable"), False, "Metal promotable")
    return {
        "shader_count": document["shader_count"],
        "metallib_sha256": document["metallib_sha256"],
        "build_log_sha256": document["build_log_sha256"],
    }


def _prepare_output_path(output_path: Path, repo_root: Path) -> None:
    historical = repo_root / "docs/sessions/20260718-metal-model-runtime/candidate-manifest.json"
    is_historical = output_path.resolve() == historical.resolve()
    if output_path.name == "candidate-manifest.json" or is_historical:
        raise CandidateRebindError("output must not overwrite the historical candidate manifest")
    if output_path.exists() or output_path.is_symlink():
        raise CandidateRebindError("output must not already exist")
    if not output_path.parent.is_dir():
        raise CandidateRebindError("output parent directory must already exist")


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _input_descriptor(snapshot: InputSnapshot, repo_root: Path) -> dict[str, str]:
    return {"path": _display_path(snapshot.path, repo_root), "sha256": snapshot.sha256}


def prepare_rebind(
    evidence_path: Path, metal_path: Path, output_path: Path, repo_root: Path
) -> dict[str, Any]:
    """Validate immutable inputs and write a non-promotable review record."""

    _prepare_output_path(output_path, repo_root)
    current_head = _current_head(repo_root)
    _require_clean_worktree(repo_root)
    current_branch = _current_branch(repo_root)
    evidence = load_json_snapshot(evidence_path, "Harbor evidence")
    metal = load_json_snapshot(metal_path, "Metal provenance")
    evidence_summary = _validate_evidence(
        evidence.document, current_head, current_branch, repo_root
    )
    metal_summary = _validate_metal(metal.document, current_head)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "candidate": {
            "repository": repo_root.name,
            "branch": current_branch,
            "head": current_head,
        },
        "evidence": {
            "artifact": _display_path(evidence.path, repo_root),
            "file_sha256": evidence.sha256,
            "canonical_sha256": evidence.document["integrity"]["canonical_sha256"],
            "input": _input_descriptor(evidence, repo_root),
            **evidence_summary,
        },
        "metal_compile_provenance": {
            "artifact": _display_path(metal.path, repo_root),
            "file_sha256": metal.sha256,
            "input": _input_descriptor(metal, repo_root),
            **metal_summary,
        },
        "promotion": {
            "verdict": "review_required",
            "promotable": False,
            "remaining_gates": ["independent_local_promotion_review"],
        },
    }
    record["integrity"] = {"canonical_sha256": _canonical_digest(record)}
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        raise CandidateRebindError("cannot write immutable review record") from exc
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harbor-envelope", required=True, type=Path)
    parser.add_argument("--metal-provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        record = prepare_rebind(
            args.harbor_envelope, args.metal_provenance, args.output, args.repo_root
        )
    except CandidateRebindError as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
