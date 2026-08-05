"""Tests for the read-only candidate-manifest promotion gate."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import pytest

from scripts.verify_candidate_manifest import CandidateManifestError, verify_candidate


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "docs/sessions/20260718-metal-model-runtime/candidate-manifest.json"


def _with_recomputed_integrity(document: dict) -> dict:
    payload = {key: value for key, value in document.items() if key != "integrity"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    document["integrity"]["canonical_sha256"] = hashlib.sha256(encoded).hexdigest()
    return document


def test_current_manifest_is_blocked_when_production_paths_drift() -> None:
    report = verify_candidate(MANIFEST, ROOT)

    assert report["schema_version"] == "pheno.candidate-manifest-review.v1"
    assert report["integrity_valid"] is True
    assert report["exact_head"] is False
    assert report["source_head_compatible"] is False
    assert "scripts/evals/run_via_harbor.sh" in report["post_head_changed_paths"]
    assert report["head_compatibility"] == "mismatch"
    assert report["workload_executed"] is False
    assert report["promotable"] is False
    assert report["status"] == "blocked"
    assert "candidate source head is not compatible with current repository HEAD" in report["reasons"]
    assert "candidate has no executed workload evidence" in report["reasons"]


def test_verifier_rejects_tampered_manifest(tmp_path: Path) -> None:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    document["candidate"]["evidence_complete"] = True
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    report = verify_candidate(path, ROOT)

    assert report["integrity_valid"] is False
    assert report["promotable"] is False
    assert report["status"] == "blocked"
    assert "manifest canonical SHA-256 does not match" in report["reasons"]


def test_verifier_rejects_duplicate_json_members(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"candidate": {}, "candidate": {}}', encoding="utf-8")

    with pytest.raises(CandidateManifestError):
        verify_candidate(path, ROOT)


def test_verifier_rejects_nonfinite_json_constants(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text('{"candidate": {"score": NaN}}', encoding="utf-8")

    with pytest.raises(CandidateManifestError):
        verify_candidate(path, ROOT)


def test_verifier_rejects_candidate_branch_mismatch(tmp_path: Path) -> None:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    document["candidate"]["branch"] = "untrusted-branch"
    path = tmp_path / "branch-mismatch.json"
    path.write_text(json.dumps(_with_recomputed_integrity(document)), encoding="utf-8")

    report = verify_candidate(path, ROOT)

    assert "candidate branch does not match repository branch" in report["reasons"]


def test_verifier_rejects_unbound_metal_provenance(tmp_path: Path) -> None:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    document["verification"]["metal_compile_provenance"]["candidate_source_head"] = "0" * 40
    path = tmp_path / "metal-mismatch.json"
    path.write_text(json.dumps(_with_recomputed_integrity(document)), encoding="utf-8")

    report = verify_candidate(path, ROOT)

    assert "Metal artifact candidate source head does not match candidate" in report["reasons"]


def test_verifier_fails_closed_without_no_follow_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(MANIFEST.read_bytes())
    monkeypatch.delattr(os, "O_NOFOLLOW")

    with pytest.raises(CandidateManifestError, match="no-follow"):
        verify_candidate(path, ROOT)
