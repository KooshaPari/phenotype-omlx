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
NIAH_MATRIX = ROOT / "research/baselines/qwen35-niah-20260727-4k8k.json"
NIAH_16K = ROOT / "research/baselines/qwen35-niah-20260727-16k-paired.json"
PROVENANCE = ROOT / (
    "docs/sessions/20260718-metal-model-runtime/artifacts/"
    "candidate-provenance-20260730.json"
)


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


def test_current_head_provenance_is_explicitly_non_promotable() -> None:
    document = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    candidate = document["candidate"]
    assert document["evidence_label"] == "provenance_only"
    assert candidate["branch"] == "feat/diffusion-trajectory-state"
    assert candidate["head"] == candidate["provenance_commit"]
    assert len(candidate["head"]) == 40
    assert candidate["evidence_complete"] is False
    assert document["artifacts"]["metallib_manifest"]["status"] == "compile_only"
    assert document["artifacts"]["metallib_manifest"]["sha256"]
    assert document["artifacts"]["device_fingerprint"]["status"] == "unknown"
    assert document["qwen35_harbor"]["status"] == "pending"
    assert document["promotion"]["verdict"] == "blocked"


def test_current_manifest_is_exact_head_and_holds_runtime_gate() -> None:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert document["candidate"]["head"] == "00a489846ebce282c69e1623ee41f9923169c08f"
    assert document["candidate"]["freeze_status"] == "current-head-integrity-reviewed"
    assert document["candidate"]["evidence_complete"] is False
    assert document["verification"]["workload_executed"] is False
    assert document["promotion"]["verdict"] == "blocked"
    assert "authorized Qwen3.5 Harbor/device evidence at current HEAD" in document["promotion"]["remaining_gates"]
    assert document["integrity"]["canonical_sha256"] == (
        "fef5b00a5aa47b3e7ed8621ca5e8a895343cac3d6d75235c853416cd76fd032b"
    )


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


def test_qwen35_native_matrix_has_real_turbo_rows() -> None:
    document = json.loads(NIAH_MATRIX.read_text(encoding="utf-8"))
    assert document["model"].lower().find("qwen3.5") >= 0
    assert document["lengths"] == [4096, 8192]
    rows = document["results"]
    assert len(rows) == 6
    turbo = [row for row in rows if row["mode"].startswith("turbo_")]
    assert len(turbo) == 4
    assert all(row["turbo_layers"] == 6 for row in turbo)
    assert all(row["compression_effective"] for row in turbo)


def test_qwen35_16k_paired_matrix_has_byte_reduction() -> None:
    document = json.loads(NIAH_16K.read_text(encoding="utf-8"))
    assert document["lengths"] == [16384]
    rows = {row["mode"]: row for row in document["results"]}
    assert rows["baseline_fp16"]["actual_len"] == 16384
    assert rows["turbo_asymmetric"]["actual_len"] == 16384
    assert rows["turbo_asymmetric"]["compression_effective"] is True
    assert rows["turbo_asymmetric"]["byte_reduction_effective"] is True
