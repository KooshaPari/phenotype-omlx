#!/usr/bin/env python3
"""Convert a local-only Harbor job into a validated EvalReport v1.0 artifact.

This adapter deliberately records no hosted-observability identity.  It consumes
Harbor's immutable ``result.json`` output and writes a sibling report with
explicit local-only provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
HARBOR_JOB_CONFIG = REPO_ROOT / "evals" / "harbor" / "jobs" / "niah-qwen35.yaml"
MODEL_CONFIG = REPO_ROOT / "config" / "smoke_models.json"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "evals" / "harbor"))

from harbor_to_cockpit import convert_job
from interchange.loader import load_report_from_dict


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_provenance(path: Path, name: str) -> dict[str, str]:
    """Bind a required regular source file by canonical path and SHA-256."""
    try:
        if not path.is_file() or path.is_symlink():
            raise OSError("not a regular file")
        resolved = path.resolve(strict=True)
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(
            f"required provenance source unavailable: {name}"
        ) from exc
    return {"path": str(resolved), "sha256": digest}


def _write_report_once(output: Path, report: dict[str, Any]) -> None:
    """Atomically create one report without overwriting or following a target."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(f"output already exists: {output}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _top_level_hash(doc: dict[str, Any]) -> str:
    payload = {key: value for key, value in doc.items() if key != "hash_chain"}
    return _sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


def _qwen35_only(model: str) -> None:
    if "qwen3.5" not in model.lower():
        raise ValueError("local Harbor provenance requires a Qwen3.5 model")


def _source_head_only(commit_sha: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
        raise ValueError("source_head must be a 40-character lowercase Git SHA")


def resolve_harbor_job_dir(output_path: Path) -> Path:
    """Resolve one completed Harbor job directory from an output root.

    Harbor writes a timestamped job directory beneath the path supplied with
    ``harbor run -o``.  Accept an already-resolved job directory as well, but
    never recursively search: a root with multiple completed jobs is ambiguous
    and must be selected explicitly by the operator.
    """
    direct_result = output_path / "result.json"
    if direct_result.is_file():
        return output_path

    candidates = (
        sorted(
            child
            for child in output_path.iterdir()
            if child.is_dir() and (child / "result.json").is_file()
        )
        if output_path.is_dir()
        else []
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"Harbor result.json not found below: {output_path}")
    names = ", ".join(candidate.name for candidate in candidates)
    raise ValueError(f"multiple Harbor job directories below {output_path}: {names}")


def _report_from_cockpit(
    cockpit: dict[str, Any],
    *,
    model: str,
    commit_sha: str,
    source_provenance: dict[str, Any],
) -> dict[str, Any]:
    cells = cockpit["cells"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        groups.setdefault(cell["suite"], []).append(cell)

    suites: list[dict[str, Any]] = []
    task_ids: list[str] = []
    for name, suite_cells in sorted(groups.items()):
        ids = [str(cell["task_id"]) for cell in suite_cells]
        task_ids.extend(ids)
        count = len(suite_cells)
        suites.append(
            {
                "suite": name,
                "n": count,
                "passed": sum(bool(cell.get("ok")) for cell in suite_cells),
                "pass_at_1": round(
                    sum(float(cell.get("pass_at_1", 0.0)) for cell in suite_cells)
                    / count,
                    4,
                ),
                "evidence_label": "live_verified",
                "task_ids": ids,
            }
        )

    total = len(cells)
    started_at = cells[0].get("created_at") or ""
    doc: dict[str, Any] = {
        "contract_version": "1.0",
        "artifact_kind": "EvaluationReport",
        "producer": {
            "name": "harbor_local_provenance",
            "version": "1.0.0",
            "commit_sha": commit_sha,
        },
        "run": {
            "run_id": "local-" + _sha256("\n".join(sorted(task_ids)))[:16],
            "started_at": started_at,
            "model": model,
            "variant": "ours",
            "judge_mode": "deterministic",
        },
        "suites": suites,
        "totals": {
            "cells": total,
            "passed": sum(bool(cell.get("ok")) for cell in cells),
            "pass_at_1": round(sum(bool(cell.get("ok")) for cell in cells) / total, 4),
        },
        "telemetry": {"mode": "local_only", "remote_exported": False},
        "source_provenance": source_provenance,
        "hash_chain": {
            "top_level_sha256": "",
            "task_ids_sorted_sha256": _sha256("\n".join(sorted(task_ids))),
        },
    }
    doc["hash_chain"]["top_level_sha256"] = _top_level_hash(doc)
    return doc


def convert_local_harbor_run(
    run_dir: Path, *, model: str, commit_sha: str
) -> tuple[dict[str, Any], Any]:
    """Convert a completed local Harbor job and validate its EvalReport."""
    _qwen35_only(model)
    _source_head_only(commit_sha)
    job_dir = resolve_harbor_job_dir(run_dir)
    source_provenance = {
        "source_head": commit_sha,
        "result_json": _file_provenance(job_dir / "result.json", "result.json"),
        "job_yaml": _file_provenance(HARBOR_JOB_CONFIG, "Harbor job YAML"),
        "model_config": _file_provenance(MODEL_CONFIG, "model config"),
    }
    report = _report_from_cockpit(
        convert_job(job_dir),
        model=model,
        commit_sha=commit_sha,
        source_provenance=source_provenance,
    )
    _, validation = load_report_from_dict(report)
    if not validation.valid:
        raise ValueError("invalid local EvalReport: " + "; ".join(validation.errors))
    return report, validation


def _commit_sha() -> str:
    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        raise ValueError("source_head cannot be resolved from the repository") from None
    _source_head_only(commit_sha)
    return commit_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report, validation = convert_local_harbor_run(
            args.run_dir, model=args.model, commit_sha=_commit_sha()
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    try:
        _write_report_once(args.output, report)
    except FileExistsError as exc:
        parser.error(str(exc))
    print(
        f"validated local EvalReport: {args.output} (warnings={len(validation.warnings)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
