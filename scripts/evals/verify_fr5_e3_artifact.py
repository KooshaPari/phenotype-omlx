#!/usr/bin/env python3
"""Fail closed verifier for the immutable Qwen3.5 FR-5 E3 promotion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
ARTIFACT_ROOT = ROOT / "docs/sessions/20260718-metal-model-runtime/artifacts"
ENVELOPE = ARTIFACT_ROOT / "qwen35-e3-runtime-state-20260726T104747Z.envelope.json"
E3_DIR = ARTIFACT_ROOT / "qwen35-e3-runtime-state-20260726T104747Z"
E3_SHA = "c4c976f45d464d7cbf215ab9325f002bdf6c6246be44465dc3ae1d41b7b6b4eb"
MANIFEST_SHA = "e6e3cab8e72fc5ff9103b9d11fc309356ce0d3267a0699b3b1e5decd1cbf9f62"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(document: dict) -> str:
    payload = {key: value for key, value in document.items() if key != "integrity"}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def verify() -> None:
    document = json.loads(ENVELOPE.read_text(encoding="utf-8"))
    assert document["evidence_label"] == "live_verified"
    assert document["synthetic"] is False
    assert document["model"] == "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
    assert document["e3"] == {
        "paired_prefill": True,
        "prefill_prompt_tokens": 512,
        "full_attention_layers": [3, 7, 11, 15, 19, 23],
        "fp16_baseline_bytes": 9437184,
        "resident_state_bytes": 1625184,
        "packed_state_bytes": 1625184,
        "byte_reduction": 0.827789306640625,
        "state_contract": "qwen35_full_attention_turbokv_post_prefill_v1",
        "qualifying": True,
    }
    assert digest(E3_DIR / "source-e3-envelope.json") == E3_SHA
    assert digest(E3_DIR / "source-e3-manifest.json") == MANIFEST_SHA
    source = json.loads((E3_DIR / "source-e3-manifest.json").read_text(encoding="utf-8"))
    assert source["measurement"]["envelope_sha256"] == E3_SHA
    assert document["e4_link"]["telemetry"] == {"mode": "local_only", "remote_exported": False}
    assert document["e4_link"]["release_evidence"] is False
    assert document["release_gate_boundaries"]["release_eligible"] is False
    for artifact in document["artifacts"]:
        path = ARTIFACT_ROOT / artifact["path"]
        assert path.is_file(), path
        assert digest(path) == artifact["sha256"], path
    assert document["integrity"]["canonical_sha256"] == canonical(document)


if __name__ == "__main__":
    verify()
    print("FR-5 E3 artifact integrity: OK")
