"""Guardrails for the current-head candidate review artifact."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).parents[2]
REVIEW = ROOT / "docs/sessions/20260718-metal-model-runtime/artifacts/candidate-review-20260726.json"
MANIFEST = ROOT / "docs/sessions/20260718-metal-model-runtime/candidate-manifest.json"
HARBOR_LATEST = ROOT / "docs/sessions/20260718-metal-model-runtime/artifacts/harbor-qwen35-20260727.json"
HARBOR_8192 = ROOT / "docs/sessions/20260718-metal-model-runtime/artifacts/harbor-qwen35-20260727-8192.json"


def test_review_preserves_stale_manifest_and_blocks_promotion() -> None:
    document = json.loads(REVIEW.read_text(encoding="utf-8"))
    assert document["evidence_label"] == "reviewed"
    assert document["candidate"]["manifest_is_stale"] is True
    assert document["review"]["verdict"] == "blocked"
    assert document["review"]["stale_manifest_untouched"] is True
    assert document["candidate"]["manifest_head"] != document["candidate"]["current_head"]


def test_review_records_all_native_evidence() -> None:
    tests = json.loads(REVIEW.read_text(encoding="utf-8"))["independent_evidence"]["current_native_tests"]
    assert set(tests) == {"c", "zig", "go", "nim", "mojo"}
    assert all("passed" in result for result in tests.values())


def test_review_canonical_digest_matches() -> None:
    document = json.loads(REVIEW.read_text(encoding="utf-8"))
    payload = {key: value for key, value in document.items() if key != "integrity"}
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    assert document["integrity"]["canonical_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_current_manifest_is_exact_head_and_holds_8192_gate() -> None:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert document["candidate"]["head"] == "25f0df6b87d0aaf0aab09e492fd2d1b5eb460e30"
    assert document["candidate"]["freeze_status"] == "current-head-reviewed"
    assert document["promotion"]["verdict"] == "review"
    assert "authorized Qwen3.5 8192-token Harbor run" not in document["promotion"]["remaining_gates"]
    payload = {key: value for key, value in document.items() if key != "integrity"}
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    assert document["integrity"]["canonical_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_latest_harbor_evidence_is_positive_but_not_8192_completion() -> None:
    document = json.loads(HARBOR_LATEST.read_text(encoding="utf-8"))
    assert document["evidence_label"] == "live_verified"
    assert document["harbor"]["reward"] == 1.0
    assert document["harbor"]["n_errors"] == 0
    assert "8192-token" in " ".join(document["promotion"]["remaining_gates"])


def test_8192_harbor_evidence_is_exact_and_positive() -> None:
    document = json.loads(HARBOR_8192.read_text(encoding="utf-8"))
    assert document["evidence_label"] == "live_verified"
    assert document["harbor"]["reward"] == 1.0
    assert document["harbor"]["prompt_tokens"] == 8192
    assert document["harbor"]["context_tokens_exact"] is True
    assert document["harbor"]["thinking_enabled"] is False
