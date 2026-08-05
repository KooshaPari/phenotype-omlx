#!/usr/bin/env python3
"""Read-only, fail-closed verification of a candidate promotion manifest.

This tool compares the recorded candidate commit with the checkout's current
HEAD and validates the manifest's canonical digest.  It never compiles,
loads, downloads, launches, benchmarks, or promotes anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping


SCHEMA_VERSION = "pheno.candidate-manifest-review.v1"
MANIFEST_SCHEMA_VERSION = "0.1"
_SHA256_LENGTH = 64
_COMMIT_LENGTH = 40


class CandidateManifestError(ValueError):
    """Raised when a candidate manifest cannot be verified safely."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object members before canonicalization.

    JSON permits parsers to choose a policy for duplicate members, but a
    promotion manifest must have one unambiguous byte representation. A
    last-write-wins parser could otherwise make the reviewed document differ
    from the canonical payload.
    """

    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise CandidateManifestError(f"duplicate manifest key: {key}")
        document[key] = value
    return document


def _reject_nonfinite(value: str) -> Any:
    """Reject JSON extensions that are not representable in canonical JSON."""

    raise CandidateManifestError(f"non-finite JSON constant is not allowed: {value}")


def _canonical_digest(document: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "integrity"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_manifest_text(path: Path) -> str:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise CandidateManifestError("manifest requires no-follow filesystem support")
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
    except FileNotFoundError as exc:
        raise CandidateManifestError("manifest must be a regular file") from exc
    except OSError as exc:
        raise CandidateManifestError("manifest must be a regular file") from exc
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise CandidateManifestError("manifest must be a regular file")
            return handle.read()
    except OSError as exc:
        raise CandidateManifestError("cannot read manifest") from exc


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_manifest_text(path),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except CandidateManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateManifestError("manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CandidateManifestError("manifest root must be an object")
    return value


def _git_head(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CandidateManifestError("cannot read repository HEAD") from exc
    head = completed.stdout.strip()
    if len(head) != _COMMIT_LENGTH or any(ch not in "0123456789abcdef" for ch in head):
        raise CandidateManifestError("repository HEAD is not a full commit SHA")
    return head


def _git_branch(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CandidateManifestError("cannot read repository branch") from exc
    branch = completed.stdout.strip()
    if not branch:
        raise CandidateManifestError("repository branch is detached or empty")
    return branch


def _metal_provenance_reasons(
    metal: Mapping[str, Any], candidate_head: Any, repo_root: Path
) -> list[str]:
    reasons: list[str] = []
    if metal.get("candidate_source_head") != candidate_head:
        reasons.append("Metal artifact candidate source head does not match candidate")
    artifact_path = metal.get("artifact")
    if not isinstance(artifact_path, str) or not artifact_path:
        return ["Metal artifact provenance path is missing"]
    relative = Path(artifact_path)
    if relative.is_absolute() or ".." in relative.parts:
        return ["Metal artifact provenance path must be repository-relative"]
    try:
        artifact = _load_manifest(repo_root / relative)
    except CandidateManifestError:
        return ["Metal artifact provenance cannot be read"]
    if artifact.get("schema_version") != "pheno.metal-compile-provenance.v1":
        reasons.append("Metal artifact provenance schema is unsupported")
    if artifact.get("candidate_source_head") != candidate_head:
        reasons.append("Metal artifact candidate source head does not match candidate")
    artifact_commit = metal.get("artifact_commit")
    if artifact.get("build_checkout_head") != artifact_commit:
        reasons.append("Metal artifact build checkout head does not match manifest")
    if artifact.get("build_checkout_head") != metal.get("head"):
        reasons.append("Metal artifact build checkout head does not match build head")
    if artifact.get("metallib_sha256") != metal.get("metallib_sha256"):
        reasons.append("Metal artifact metallib digest does not match manifest")
    if artifact.get("build_log_sha256") != metal.get("build_log_sha256"):
        reasons.append("Metal artifact build-log digest does not match manifest")
    if artifact.get("shader_count") != metal.get("shader_count"):
        reasons.append("Metal artifact shader count does not match manifest")
    if artifact.get("source_head_compatible") is not True:
        reasons.append("Metal artifact does not declare source-head compatibility")
    return reasons


def _source_head_compatibility(
    manifest_head: Any,
    current_head: str,
    repo_root: Path,
    protected_paths: Any,
) -> tuple[bool, list[str]]:
    """Allow only bookkeeping commits after the evaluated source head.

    A tracked manifest and its verifier cannot be written without changing
    repository HEAD.  The source candidate remains compatible when every
    committed path after its recorded head avoids the declared production
    paths; any runtime/source-file drift fails closed.
    """

    if not isinstance(manifest_head, str) or len(manifest_head) != _COMMIT_LENGTH:
        return False, []
    if manifest_head == current_head:
        return True, []
    if not isinstance(protected_paths, list) or not protected_paths:
        return False, []
    roots = [path.strip("/") for path in protected_paths if isinstance(path, str) and path.strip("/")]
    if len(roots) != len(protected_paths):
        return False, []
    try:
        ancestor = subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", manifest_head, current_head],
            capture_output=True,
            timeout=10,
        )
        if ancestor.returncode != 0:
            return False, []
        changed = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--name-only", f"{manifest_head}..{current_head}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False, []
    paths = [line for line in changed.stdout.splitlines() if line]

    def touches_protected(path: str) -> bool:
        return any(path == root or path.startswith(f"{root}/") for root in roots)

    return bool(paths) and not any(touches_protected(path) for path in paths), paths


def verify_candidate(manifest_path: Path, repo_root: Path) -> dict[str, Any]:
    """Return a provenance report without authorizing any execution."""

    document = _load_manifest(manifest_path)
    current_head = _git_head(repo_root)
    current_branch = _git_branch(repo_root)
    if document.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise CandidateManifestError("unsupported candidate manifest schema_version")
    candidate = document.get("candidate")
    verification = document.get("verification")
    metal = verification.get("metal_compile_provenance", {}) if isinstance(verification, Mapping) else {}
    if not isinstance(candidate, Mapping) or not isinstance(verification, Mapping):
        raise CandidateManifestError("manifest candidate and verification objects are required")
    if not isinstance(metal, Mapping):
        raise CandidateManifestError("metal_compile_provenance must be an object")

    identity_reasons: list[str] = []
    if candidate.get("repository") != repo_root.name:
        identity_reasons.append("candidate repository does not match repository root")
    if candidate.get("branch") != current_branch:
        identity_reasons.append("candidate branch does not match repository branch")

    manifest_head = candidate.get("head")
    exact_head = manifest_head == current_head
    source_head_compatible, post_head_changed_paths = _source_head_compatibility(
        manifest_head,
        current_head,
        repo_root,
        document.get("changes", {}).get("production_paths")
        if isinstance(document.get("changes"), Mapping)
        else None,
    )
    declared_digest = None
    integrity = document.get("integrity")
    if isinstance(integrity, Mapping):
        declared_digest = integrity.get("canonical_sha256")
    try:
        computed_digest = _canonical_digest(document)
    except (TypeError, ValueError) as exc:
        raise CandidateManifestError("manifest contains values that cannot be canonicalized") from exc
    integrity_valid = declared_digest == computed_digest

    workload_executed = verification.get("workload_executed") is True
    evidence_complete = candidate.get("evidence_complete") is True
    artifact_reasons = _metal_provenance_reasons(metal, manifest_head, repo_root)
    artifact_bound = metal.get("artifact_bound_to_candidate_head") is True and not artifact_reasons
    promotion_verdict = document.get("promotion", {}).get("verdict") if isinstance(document.get("promotion"), Mapping) else None

    reasons: list[str] = []
    reasons.extend(identity_reasons)
    reasons.extend(artifact_reasons)
    if not integrity_valid:
        reasons.append("manifest canonical SHA-256 does not match")
    if not source_head_compatible:
        reasons.append("candidate source head is not compatible with current repository HEAD")
    if not workload_executed:
        reasons.append("candidate has no executed workload evidence")
    if not evidence_complete:
        reasons.append("candidate evidence is incomplete")
    if not artifact_bound:
        reasons.append("Metal artifact is not bound to the candidate head")
    if promotion_verdict != "accepted":
        reasons.append("manifest promotion verdict is not accepted")

    promotable = not reasons
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "current_head": current_head,
        "manifest_head": manifest_head,
        "exact_head": exact_head,
        "source_head_compatible": source_head_compatible,
        "head_compatibility": "exact" if exact_head else (
            "bookkeeping_commits" if source_head_compatible else "mismatch"
        ),
        "post_head_changed_paths": post_head_changed_paths,
        "integrity_valid": integrity_valid,
        "workload_executed": workload_executed,
        "evidence_complete": evidence_complete,
        "artifact_bound_to_candidate_head": artifact_bound,
        "promotion_verdict": promotion_verdict,
        "promotable": promotable,
        "status": "accepted" if promotable else "blocked",
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args(argv)
    try:
        report = verify_candidate(args.manifest, args.repo_root)
    except CandidateManifestError as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["promotable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
