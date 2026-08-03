#!/usr/bin/env python3
"""Prepare immutable current-head evidence for an independent promotion review.

The resulting record is deliberately non-promotable. It binds a fresh Qwen3.5
Harbor envelope and a fresh Metal compile provenance record to the current
checkout without changing a historical candidate manifest or running work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


SCHEMA_VERSION = "pheno.candidate-rebind-review.v1"
_SHA256_LENGTH = 64
_COMMIT_LENGTH = 40


class CandidateRebindError(ValueError):
    """Raised when evidence cannot safely be prepared for review."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise CandidateRebindError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _reject_nonfinite(value: str) -> Any:
    raise CandidateRebindError(f"non-finite JSON constant is not allowed: {value}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CandidateRebindError(f"{label} must be a regular file")
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, CandidateRebindError) as exc:
        raise CandidateRebindError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise CandidateRebindError(f"{label} root must be an object")
    return document


def _canonical_digest(document: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "integrity"}
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CandidateRebindError("record cannot be canonicalized") from exc
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


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


def _validate_evidence(document: Mapping[str, Any], current_head: str) -> dict[str, Any]:
    _require_exact(document.get("schema_version"), "0.1", "evidence schema_version")
    _require_exact(document.get("evidence_label"), "live_verified", "evidence_label")
    declared_digest = _mapping(document.get("integrity"), "evidence integrity").get(
        "canonical_sha256"
    )
    if not _is_sha256(declared_digest) or declared_digest != _canonical_digest(document):
        raise CandidateRebindError("evidence canonical SHA-256 does not match")

    model = document.get("model")
    if not isinstance(model, str) or "qwen3.5" not in model.lower() or "qwen2.5" in model.lower():
        raise CandidateRebindError("evidence model must be an allowlisted Qwen3.5 model")

    candidate = _mapping(document.get("candidate"), "evidence candidate")
    _require_exact(candidate.get("source_head"), current_head, "evidence source head vs current repository HEAD")
    harbor = _mapping(document.get("harbor"), "Harbor evidence")
    for field in ("job_id", "trial_name", "prompt_sha256"):
        if not isinstance(harbor.get(field), str) or not harbor[field]:
            raise CandidateRebindError(f"Harbor {field} must be non-empty")
    if not _is_sha256(harbor["prompt_sha256"]):
        raise CandidateRebindError("Harbor prompt_sha256 must be a lowercase SHA-256")
    _require_exact(harbor.get("environment"), "apple-container", "Harbor environment")
    _require_integer(harbor.get("n_errors"), 0, "Harbor n_errors")
    _require_integer(harbor.get("n_retries"), 0, "Harbor retries")
    _require_exact(harbor.get("fallback_applied"), False, "Harbor fallback_applied")
    _require_score(harbor.get("reward"), 1.0, "Harbor reward")
    _require_score(harbor.get("pass_at_1"), 1.0, "Harbor pass_at_1")
    _require_integer(harbor.get("requested_context_tokens"), 8192, "Harbor requested_context_tokens")
    _require_integer(harbor.get("prompt_tokens"), 8192, "Harbor prompt_tokens")
    _require_exact(harbor.get("context_tokens_exact"), True, "Harbor context_tokens_exact")

    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CandidateRebindError("evidence artifacts must be a non-empty array")
    for artifact in artifacts:
        item = _mapping(artifact, "evidence artifact")
        if not isinstance(item.get("path"), str) or not item["path"]:
            raise CandidateRebindError("evidence artifact path must be non-empty")
        if not _is_sha256(item.get("sha256")):
            raise CandidateRebindError("evidence artifact sha256 must be a lowercase SHA-256")

    return {"model": model, "harbor": dict(harbor), "artifact_count": len(artifacts)}


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
    _require_exact(document.get("source_head_compatible"), True, "Metal source_head_compatible")
    _require_exact(document.get("shader_count"), 20, "Metal shader_count")
    for field in ("metallib_sha256", "build_log_sha256"):
        if not _is_sha256(document.get(field)):
            raise CandidateRebindError(f"Metal {field} must be a lowercase SHA-256")
    _require_exact(document.get("workload_executed"), False, "Metal workload_executed")
    _require_exact(document.get("device_dispatch_executed"), False, "Metal device_dispatch_executed")
    _require_exact(document.get("model_loaded"), False, "Metal model_loaded")
    _require_exact(document.get("promotable"), False, "Metal promotable")
    return {
        "shader_count": document["shader_count"],
        "metallib_sha256": document["metallib_sha256"],
        "build_log_sha256": document["build_log_sha256"],
    }


def _prepare_output_path(output_path: Path, repo_root: Path) -> None:
    historical = repo_root / "docs/sessions/20260718-metal-model-runtime/candidate-manifest.json"
    if output_path.name == "candidate-manifest.json" or output_path.resolve() == historical.resolve():
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


def prepare_rebind(
    evidence_path: Path, metal_path: Path, output_path: Path, repo_root: Path
) -> dict[str, Any]:
    """Validate immutable inputs and write a non-promotable review record."""

    _prepare_output_path(output_path, repo_root)
    current_head = _current_head(repo_root)
    _require_clean_worktree(repo_root)
    evidence = _load_json(evidence_path, "Harbor evidence")
    metal = _load_json(metal_path, "Metal provenance")
    evidence_summary = _validate_evidence(evidence, current_head)
    metal_summary = _validate_metal(metal, current_head)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "candidate": {
            "repository": repo_root.name,
            "branch": _current_branch(repo_root),
            "head": current_head,
        },
        "evidence": {
            "artifact": _display_path(evidence_path, repo_root),
            "file_sha256": _file_digest(evidence_path),
            "canonical_sha256": evidence["integrity"]["canonical_sha256"],
            **evidence_summary,
        },
        "metal_compile_provenance": {
            "artifact": _display_path(metal_path, repo_root),
            "file_sha256": _file_digest(metal_path),
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
