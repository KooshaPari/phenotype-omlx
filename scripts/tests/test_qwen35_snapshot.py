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


def _write_safetensors(path: Path, payload: bytes) -> int:
    header = json.dumps(
        {"tensor": {"dtype": "U8", "shape": [len(payload)], "data_offsets": [0, len(payload)]}},
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)
    return len(payload)


def test_index_scope_mismatch_fails_closed(tmp_path: Path) -> None:
    model_payload = _write_safetensors(tmp_path / "model.safetensors", b"weights")
    vision = tmp_path / "vision.safetensors"
    vision_payload = _write_safetensors(vision, b"vision")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": model_payload},
                "weight_map": {"x": "model.safetensors", "vision": "vision.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    report = MODULE.verify_snapshot(tmp_path, "mlx-community/Qwen3.5-0.8B-OptiQ-4bit")
    assert report["integrity"]["status"] == "failed"
    assert any("metadata_scope_mismatch" in error for error in report["integrity"]["errors"])
    assert report["snapshot"]["indexed_payload_size_bytes"] == model_payload + vision_payload
    assert report["workload_executed"] is False


def test_declared_sidecar_scope_is_verified(tmp_path: Path) -> None:
    for relative in MODULE.REQUIRED:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative not in {"model.safetensors", "model.safetensors.index.json", "config.json"}:
            path.write_text("{}" if path.suffix == ".json" else "", encoding="utf-8")
    model_payload = _write_safetensors(tmp_path / "model.safetensors", b"weights")
    vision_payload = _write_safetensors(tmp_path / "vision.safetensors", b"vision")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "optiq_vision": {"sidecar": "vision.safetensors"}}),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": model_payload},
                "weight_map": {"x": "model.safetensors", "vision": "vision.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    report = MODULE.verify_snapshot(tmp_path, "mlx-community/Qwen3.5-0.8B-OptiQ-4bit")
    assert report["integrity"]["status"] == "verified_with_sidecar_scope"
    assert report["snapshot"]["indexed_payload_size_bytes"] == model_payload + vision_payload
    assert report["snapshot"]["index_scope"] == "declared_sidecars_excluded"


def test_index_rejects_path_traversal(tmp_path: Path) -> None:
    _write_safetensors(tmp_path / "model.safetensors", b"weights")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 7}, "weight_map": {"x": "../outside"}}),
        encoding="utf-8",
    )
    report = MODULE.verify_snapshot(tmp_path, "mlx-community/Qwen3.5-0.8B-OptiQ-4bit")
    assert report["integrity"]["status"] == "failed"
    assert any("unsafe_weight_map_path" in error for error in report["integrity"]["errors"])


def test_mtp_is_optional() -> None:
    assert "optiq/mtp.safetensors" not in MODULE.REQUIRED
    assert "optiq/mtp.safetensors" in MODULE.OPTIONAL


def test_snapshot_ref_cannot_escape_cache_root(tmp_path: Path) -> None:
    (tmp_path / "outside").mkdir()
    repo = tmp_path / "models--mlx-community--Qwen3.5-0.8B-OptiQ-4bit"
    ref = repo / "refs" / "main"
    ref.parent.mkdir(parents=True)
    (repo / "snapshots").mkdir()
    ref.write_text("../../outside\n", encoding="utf-8")
    try:
        MODULE._snapshot_dir("mlx-community/Qwen3.5-0.8B-OptiQ-4bit", tmp_path)
    except (FileNotFoundError, ValueError) as exc:
        assert "snapshot" in str(exc) or "ref" in str(exc)
    else:
        raise AssertionError("snapshot ref traversal must fail closed")
