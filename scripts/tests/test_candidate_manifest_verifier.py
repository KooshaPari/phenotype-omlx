"""Tests for the read-only candidate-manifest promotion gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_candidate_manifest import CandidateManifestError, verify_candidate


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "docs/sessions/20260718-metal-model-runtime/candidate-manifest.json"


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
