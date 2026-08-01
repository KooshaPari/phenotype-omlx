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
from pathlib import Path
import subprocess
from typing import Any, Mapping


SCHEMA_VERSION = "pheno.candidate-manifest-review.v1"
_SHA256_LENGTH = 64
_COMMIT_LENGTH = 40


class CandidateManifestError(ValueError):
    """Raised when a candidate manifest cannot be verified safely."""


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


def _load_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CandidateManifestError("manifest must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
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


def verify_candidate(manifest_path: Path, repo_root: Path) -> dict[str, Any]:
    """Return a provenance report without authorizing any execution."""

    document = _load_manifest(manifest_path)
    current_head = _git_head(repo_root)
    candidate = document.get("candidate")
    verification = document.get("verification")
    metal = verification.get("metal_compile_provenance", {}) if isinstance(verification, Mapping) else {}
    if not isinstance(candidate, Mapping) or not isinstance(verification, Mapping):
        raise CandidateManifestError("manifest candidate and verification objects are required")
    if not isinstance(metal, Mapping):
        raise CandidateManifestError("metal_compile_provenance must be an object")

    manifest_head = candidate.get("head")
    exact_head = manifest_head == current_head
    declared_digest = None
    integrity = document.get("integrity")
    if isinstance(integrity, Mapping):
        declared_digest = integrity.get("canonical_sha256")
    integrity_valid = declared_digest == _canonical_digest(document)

    workload_executed = verification.get("workload_executed") is True
    evidence_complete = candidate.get("evidence_complete") is True
    artifact_bound = metal.get("artifact_bound_to_candidate_head") is True
    promotion_verdict = document.get("promotion", {}).get("verdict") if isinstance(document.get("promotion"), Mapping) else None

    reasons: list[str] = []
    if not integrity_valid:
        reasons.append("manifest canonical SHA-256 does not match")
    if not exact_head:
        reasons.append("candidate head does not match current repository HEAD")
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
