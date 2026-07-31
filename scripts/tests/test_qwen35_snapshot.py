"""Fail-closed tests for the Qwen3.5 snapshot integrity gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "verify_qwen35_snapshot", ROOT / "scripts" / "verify_qwen35_snapshot.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_index_size_mismatch_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 99}, "weight_map": {"x": "model.safetensors"}}),
        encoding="utf-8",
    )
    report = MODULE.verify_snapshot(tmp_path, "mlx-community/Qwen3.5-0.8B-OptiQ-4bit")
    assert report["integrity"]["status"] == "failed"
    assert any("size_mismatch" in error for error in report["integrity"]["errors"])
    assert report["workload_executed"] is False
