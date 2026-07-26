"""Integrity and claim-boundary tests for the committed local Harbor evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
ENVELOPE = (
    ROOT
    / "docs/sessions/20260718-metal-model-runtime/artifacts"
    / "harbor-qwen35-local-20260726T061831Z.envelope.json"
)


def _canonical_without_integrity(document: dict) -> bytes:
    payload = {key: value for key, value in document.items() if key != "integrity"}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def test_local_harbor_evidence_is_hashed_qwen35_retrieval_only() -> None:
    """The copied local run is immutable provenance, not FR-5 E3 proof."""

    document = json.loads(ENVELOPE.read_text(encoding="utf-8"))
    assert document["schema_version"] == "0.2"
    assert document["evidence_label"] == "live_verified"
    assert document["evidence_scope"] == "local_only_qwen35_niah_api_retrieval"
    assert document["model"] == "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
    assert document["candidate"]["endpoint_scope"] == "dedicated_8766"
    assert document["telemetry"] == {"mode": "local_only", "remote_exported": False}
    assert document["harbor"]["environment"] == "apple-container"
    assert document["harbor"]["n_errors"] == 0
    assert document["harbor"]["reward"] == 1.0

    qualification = document["fr5_qualification"]
    assert qualification["e3_compression_qualifying"] is False
    assert qualification["e4_local_retrieval_observed"] is True
    assert qualification["release_eligible"] is False
    assert set(qualification["e3_missing_metrics"]) == {
        "packed_state_bytes",
        "fp16_baseline_bytes",
        "byte_reduction",
    }
    assert "not an FR-5 E3 compression proof" in qualification["non_claims"]

    artifact_root = ENVELOPE.parent
    for artifact in document["artifacts"]:
        path = artifact_root / artifact["path"]
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]

    copied_manifest = artifact_root / (
        "harbor-qwen35-local-20260726T061831Z/candidate-manifest.local.json"
    )
    manifest = json.loads(copied_manifest.read_text(encoding="utf-8"))
    assert manifest["candidate"]["evidence_complete"] is False
    assert manifest["candidate"]["release_eligible"] is False
    assert manifest["telemetry"] == {"mode": "local_only", "remote_exported": False}

    expected = hashlib.sha256(_canonical_without_integrity(document)).hexdigest()
    assert document["integrity"]["canonical_sha256"] == expected
