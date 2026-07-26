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
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "evals" / "harbor"))

from harbor_to_cockpit import convert_job
from interchange.loader import load_report_from_dict


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _top_level_hash(doc: dict[str, Any]) -> str:
    payload = {key: value for key, value in doc.items() if key != "hash_chain"}
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _qwen35_only(model: str) -> None:
    if "qwen3.5" not in model.lower():
        raise ValueError("local Harbor provenance requires a Qwen3.5 model")


def _report_from_cockpit(cockpit: dict[str, Any], *, model: str, commit_sha: str) -> dict[str, Any]:
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
                "pass_at_1": round(sum(float(cell.get("pass_at_1", 0.0)) for cell in suite_cells) / count, 4),
                "evidence_label": "live_verified",
                "task_ids": ids,
            }
        )

    total = len(cells)
    started_at = cells[0].get("created_at") or ""
    doc: dict[str, Any] = {
        "contract_version": "1.0",
        "artifact_kind": "EvaluationReport",
        "producer": {"name": "harbor_local_provenance", "version": "1.0.0", "commit_sha": commit_sha},
        "run": {"run_id": "local-" + _sha256("\n".join(sorted(task_ids)))[:16], "started_at": started_at,
                "model": model, "variant": "ours", "judge_mode": "deterministic"},
        "suites": suites,
        "totals": {"cells": total, "passed": sum(bool(cell.get("ok")) for cell in cells),
                   "pass_at_1": round(sum(bool(cell.get("ok")) for cell in cells) / total, 4)},
        "telemetry": {"mode": "local_only", "remote_exported": False},
        "hash_chain": {"top_level_sha256": "", "task_ids_sorted_sha256": _sha256("\n".join(sorted(task_ids)))},
    }
    doc["hash_chain"]["top_level_sha256"] = _top_level_hash(doc)
    return doc


def convert_local_harbor_run(
    run_dir: Path, *, model: str, commit_sha: str
) -> tuple[dict[str, Any], Any]:
    """Convert a completed local Harbor job and validate its EvalReport."""
    _qwen35_only(model)
    if not (run_dir / "result.json").is_file():
        raise FileNotFoundError(f"Harbor result.json not found: {run_dir / 'result.json'}")
    report = _report_from_cockpit(convert_job(run_dir), model=model, commit_sha=commit_sha)
    _, validation = load_report_from_dict(report)
    if not validation.valid:
        raise ValueError("invalid local EvalReport: " + "; ".join(validation.errors))
    return report, validation


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report, validation = convert_local_harbor_run(args.run_dir, model=args.model, commit_sha=_commit_sha())
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"validated local EvalReport: {args.output} (warnings={len(validation.warnings)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
