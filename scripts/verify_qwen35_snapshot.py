#!/usr/bin/env python3
"""Verify the local, cached Qwen3.5 OptiQ snapshot without loading the model.

This is an integrity/provenance gate only. It never downloads weights, imports MLX,
starts a server, or runs an evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

REQUIRED = (
    "config.json",
    "generation_config.json",
    "kv_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "optiq_metadata.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "optiq/mtp.safetensors",
    "optiq/optiq_vision.safetensors",
)


def _cache_root() -> Path:
    configured = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if configured:
        return Path(configured).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _snapshot_dir(model_id: str, cache_root: Path) -> Path:
    owner, name = model_id.split("/", 1)
    repo = cache_root / f"models--{owner}--{name}"
    ref = repo / "refs" / "main"
    if not ref.is_file():
        raise FileNotFoundError(f"missing cached main ref: {ref}")
    snapshot = repo / "snapshots" / ref.read_text(encoding="utf-8").strip()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"missing cached snapshot: {snapshot}")
    return snapshot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(snapshot: Path, model_id: str) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    errors: list[str] = []
    for relative in REQUIRED:
        path = snapshot / relative
        if not path.exists():
            errors.append(f"missing:{relative}")
            continue
        if not path.is_file():
            errors.append(f"not_file:{relative}")
            continue
        entries[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "resolved_path": relative,
        }

    index_path = snapshot / "model.safetensors.index.json"
    indexed_files: list[str] = []
    index_total: int | None = None
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            metadata = index.get("metadata") or {}
            index_total = metadata.get("total_size")
            weight_map = index.get("weight_map") or {}
            indexed_files = sorted(set(str(value) for value in weight_map.values()))
            actual_total = sum(
                (snapshot / relative).stat().st_size
                for relative in indexed_files
                if (snapshot / relative).is_file()
            )
            if not isinstance(index_total, int):
                errors.append("index:metadata.total_size_missing_or_non_integer")
            elif actual_total != index_total:
                errors.append(
                    f"index:size_mismatch:metadata={index_total}:actual={actual_total}"
                )
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"index:invalid:{type(exc).__name__}")
    else:
        errors.append("index:missing")

    config_type = None
    config_path = snapshot / "config.json"
    if config_path.is_file():
        try:
            config_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type")
            if config_type != "qwen3_5":
                errors.append(f"config:model_type={config_type!r}")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"config:invalid:{type(exc).__name__}")

    return {
        "schema_version": "0.1",
        "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evidence_label": "snapshot_integrity",
        "model": model_id,
        "snapshot": {
            "cache_relative": str(snapshot).replace(str(Path.home()), "$HOME"),
            "required_files": entries,
            "indexed_files": indexed_files,
            "index_total_size_bytes": index_total,
            "config_model_type": config_type,
        },
        "integrity": {"status": "verified" if not errors else "failed", "errors": errors},
        "workload_executed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from omlx_research.smoke_models import default_model_for

        model_id = default_model_for("readiness")
        if model_id != "mlx-community/Qwen3.5-0.8B-OptiQ-4bit":
            raise RuntimeError(f"unexpected readiness model: {model_id}")
        report = verify_snapshot(_snapshot_dir(model_id, _cache_root()), model_id)
    except (OSError, RuntimeError, ValueError) as exc:
        report = {
            "schema_version": "0.1",
            "evidence_label": "snapshot_integrity",
            "integrity": {"status": "failed", "errors": [str(exc)]},
            "workload_executed": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["integrity"]["status"], "output": str(args.output)}))
    return 0 if report["integrity"]["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
