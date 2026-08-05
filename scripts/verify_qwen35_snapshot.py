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
)
OPTIONAL = ("optiq/mtp.safetensors",)


def _cache_root() -> Path:
    for variable in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "SSD_HF_CACHE"):
        configured = os.environ.get(variable)
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
    if not ref.is_file() or ref.is_symlink():
        raise FileNotFoundError(f"missing cached main ref: {ref}")
    revision = ref.read_text(encoding="utf-8").strip()
    if not revision or "/" in revision or "\\" in revision:
        raise ValueError("cached main ref must contain one snapshot revision")
    snapshots_root = (repo / "snapshots").resolve()
    snapshot = snapshots_root / revision
    resolved_snapshot = snapshot.resolve(strict=False)
    try:
        resolved_snapshot.relative_to(snapshots_root)
    except ValueError as exc:
        raise ValueError("cached main ref escapes snapshots root") from exc
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise FileNotFoundError(f"missing cached snapshot: {snapshot}")
    return resolved_snapshot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_index_path(value: Any) -> str:
    """Return a safe snapshot-relative path or raise on traversal."""
    if not isinstance(value, str) or not value:
        raise ValueError("index:weight_map_path_not_string")
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"index:unsafe_weight_map_path:{value!r}")
    return "/".join(path.parts)


def _safetensors_payload_bytes(path: Path) -> int:
    """Read safetensors framing and return tensor payload bytes without dependencies."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("safetensors:short_header_length")
        header_length = int.from_bytes(prefix, "little")
        header = stream.read(header_length)
        if len(header) != header_length:
            raise ValueError("safetensors:short_header")
    try:
        metadata = json.loads(header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("safetensors:invalid_header_json") from exc
    offsets: list[tuple[int, int]] = []
    for name, tensor in metadata.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict) or not isinstance(tensor.get("data_offsets"), list):
            raise ValueError(f"safetensors:invalid_tensor:{name!r}")
        raw_offsets = tensor["data_offsets"]
        if len(raw_offsets) != 2 or not all(isinstance(item, int) for item in raw_offsets):
            raise ValueError(f"safetensors:invalid_offsets:{name!r}")
        start, end = raw_offsets
        if start < 0 or end < start:
            raise ValueError(f"safetensors:invalid_offsets:{name!r}")
        offsets.append((start, end))
    if not offsets:
        return 0
    start, end = min(item[0] for item in offsets), max(item[1] for item in offsets)
    payload_start = 8 + header_length
    if payload_start + end > path.stat().st_size:
        raise ValueError("safetensors:offsets_exceed_file")
    return end - start


def _redacted_snapshot(snapshot: Path, cache_root: Path | None) -> str:
    """Never emit an absolute home/custom-cache path in evidence."""
    if cache_root is not None:
        try:
            return "$CACHE/" + snapshot.relative_to(cache_root).as_posix()
        except ValueError:
            pass
    return "$SNAPSHOT/" + snapshot.name


def verify_snapshot(
    snapshot: Path, model_id: str, *, cache_root: Path | None = None
) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    optional_entries: dict[str, Any] = {}
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
    for relative in OPTIONAL:
        path = snapshot / relative
        if path.is_file():
            optional_entries[relative] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "resolved_path": relative,
            }

    config_type = None
    declared_sidecars: set[str] = set()
    config_path = snapshot / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config_type = config.get("model_type")
            if config_type != "qwen3_5":
                errors.append(f"config:model_type={config_type!r}")
            vision = config.get("optiq_vision") or {}
            if isinstance(vision, dict) and vision.get("sidecar") is not None:
                declared_sidecars.add(_safe_index_path(vision["sidecar"]))
        except (OSError, ValueError, TypeError) as exc:
            message = str(exc)
            errors.append(
                message if message.startswith("index:") else f"config:invalid:{type(exc).__name__}"
            )

    index_path = snapshot / "model.safetensors.index.json"
    indexed_files: list[str] = []
    index_total: int | None = None
    actual_total = 0
    payload_total = 0
    payload_sizes: dict[str, int] = {}
    index_scope = "unknown"
    warnings: list[str] = []
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            metadata = index.get("metadata") or {}
            index_total = metadata.get("total_size")
            weight_map = index.get("weight_map") or {}
            if not isinstance(weight_map, dict):
                raise ValueError("index:weight_map_not_object")
            indexed_files = sorted({_safe_index_path(value) for value in weight_map.values()})
            missing_indexed = [
                relative for relative in indexed_files if not (snapshot / relative).is_file()
            ]
            if missing_indexed:
                errors.extend(f"index:missing_weight_file:{relative}" for relative in missing_indexed)
            for relative in indexed_files:
                path = snapshot / relative
                if path.is_file() and relative not in entries:
                    entries[relative] = {
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                        "resolved_path": relative,
                    }
            actual_total = sum(
                (snapshot / relative).stat().st_size
                for relative in indexed_files
                if (snapshot / relative).is_file()
            )
            for relative in indexed_files:
                path = snapshot / relative
                if path.is_file():
                    try:
                        payload_sizes[relative] = _safetensors_payload_bytes(path)
                    except (OSError, ValueError) as exc:
                        errors.append(f"index:payload_invalid:{relative}:{exc}")
            payload_total = sum(payload_sizes.values())
            if not isinstance(index_total, int):
                errors.append("index:metadata.total_size_missing_or_non_integer")
            else:
                sidecar_payload = sum(
                    payload_sizes[relative]
                    for relative in declared_sidecars
                    if relative in payload_sizes and relative in indexed_files
                )
                scoped_payload = payload_total - sidecar_payload
                if payload_total == index_total:
                    index_scope = "all_indexed_payload"
                elif declared_sidecars and scoped_payload == index_total:
                    index_scope = "declared_sidecars_excluded"
                    warnings.append(
                        "index:metadata_scope_excludes_declared_sidecars:"
                        + ",".join(sorted(declared_sidecars))
                    )
                else:
                    index_scope = "mismatch"
                    errors.append(
                        "index:metadata_scope_mismatch:"
                        f"metadata={index_total}:indexed_payload={payload_total}:"
                        f"indexed_files_size={actual_total}"
                    )
        except (OSError, ValueError, TypeError) as exc:
            message = str(exc)
            errors.append(
                message if message.startswith("index:") else f"index:invalid:{type(exc).__name__}"
            )
    else:
        errors.append("index:missing")

    status = "verified_with_sidecar_scope" if not errors and warnings else "verified" if not errors else "failed"
    return {
        "schema_version": "0.1",
        "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evidence_label": "snapshot_integrity",
        "model": model_id,
        "snapshot": {
            "cache_relative": _redacted_snapshot(snapshot, cache_root),
            "required_files": entries,
            "optional_files": optional_entries,
            "indexed_files": indexed_files,
            "index_total_size_bytes": index_total,
            "indexed_files_size_bytes": actual_total if index_path.is_file() else None,
            "indexed_payload_size_bytes": payload_total if index_path.is_file() else None,
            "indexed_payloads_bytes": payload_sizes if index_path.is_file() else {},
            "declared_sidecars": sorted(declared_sidecars),
            "index_scope": index_scope,
            "config_model_type": config_type,
        },
        "integrity": {"status": status, "errors": errors, "warnings": warnings},
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
        cache_root = _cache_root()
        report = verify_snapshot(_snapshot_dir(model_id, cache_root), model_id, cache_root=cache_root)
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
    return 0 if report["integrity"]["status"] in {"verified", "verified_with_sidecar_scope"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
