"""CLI envelope tests for fail-closed Qwen3.5 E3 evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "niah_qwen35_e3_compression.py"


def _module():
    spec = importlib.util.spec_from_file_location("niah_qwen35_e3_compression", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nonqualifying_envelope_never_claims_live_verified() -> None:
    module = _module()
    document = module._envelope("Qwen/Qwen3.5-0.8B", 512, metrics=None, error="no cache")
    assert document["evidence_label"] == "not_applicable"
    assert document["reported"] is False
    assert document["e3_compression_qualifying"] is False
    assert document["metrics"] is None
    assert document["not_applicable_reason"] == "no cache"


def test_qualifying_envelope_requires_real_runner_metrics() -> None:
    module = _module()
    metrics = {
        "packed_state_bytes": 700,
        "fp16_baseline_bytes": 2_000,
        "resident_state_bytes": 800,
        "byte_reduction": 0.6,
        "e3_compression_qualifying": True,
    }
    document = module._envelope("Qwen/Qwen3.5-0.8B", 512, metrics=metrics, error=None)
    assert document["evidence_label"] == "live_verified"
    assert document["e3_compression_qualifying"] is True
    assert document["metrics"] is metrics
