"""Contract checks for the immutable Harbor evidence envelope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
ENVELOPE = ROOT / "docs/sessions/20260718-metal-model-runtime/artifacts/harbor-qwen35-20260726.json"


def _canonical_without_integrity(document: dict) -> bytes:
    payload = {key: value for key, value in document.items() if key != "integrity"}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def test_harbor_envelope_is_live_verified_and_current() -> None:
    document = json.loads(ENVELOPE.read_text(encoding="utf-8"))
    assert document["schema_version"] == "0.1"
    assert document["evidence_label"] == "live_verified"
    assert document["candidate"]["head"] == "c71bc6228d0c2e5b7ebb392e7e36ee620a8974f8"
    assert document["harbor"]["job_id"] == "c8e0d681-4754-4f94-8b00-7e82c92ee653"
    assert document["harbor"]["trial_name"] == "omlx-niah-api-smoke__ooX9Kjs"
    assert document["harbor"]["reward"] == 1.0
    assert document["harbor"]["n_errors"] == 0
    assert document["harbor"]["environment"] == "apple-container"
    assert document["model"] == "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
    assert len(document["artifacts"]) == 4
    for artifact in document["artifacts"]:
        digest = artifact["sha256"]
        assert len(digest) == hashlib.sha256().digest_size * 2
        assert all(character in "0123456789abcdef" for character in digest)


def test_harbor_envelope_canonical_digest_matches() -> None:
    document = json.loads(ENVELOPE.read_text(encoding="utf-8"))
    expected = hashlib.sha256(_canonical_without_integrity(document)).hexdigest()
    assert document["integrity"]["canonical_sha256"] == expected
