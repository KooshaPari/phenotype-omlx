#!/usr/bin/env python3
"""Convert Harbor result.json files to bench-cockpit cells format.

Harbor jobs produce two result.json layers:
  1. Job-level  (.runs/<job>/<ts>/result.json)
     → stats.evals mapping eval_name → metrics/reward_stats
  2. Trial-level (.runs/<job>/<ts>/<trial>/result.json)
     → per-trial verifier_result, agent_result, timing phases

bench-cockpit expects { summary: { meta, by_variant }, cells: [...] }

Usage:
    python harbor_to_cockpit.py <run_dir> [-o output.json]
    python harbor_to_cockpit.py .runs/harbor-eval-judge-resume/2026-07-22__22-39-39
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Harbor result.json types (job-level)
# ---------------------------------------------------------------------------


def load_job_result(path: Path) -> dict:
    """Load a Harbor job-level result.json."""
    with open(path) as f:
        data = json.load(f)
    if "stats" not in data or "evals" not in data["stats"]:
        raise ValueError(f"Not a Harbor job result: {path} (missing stats.evals)")
    return data


def load_trial_result(path: Path) -> dict | None:
    """Load a Harbor trial-level result.json, or None on error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Eval name parsing  (e.g. "oracle__adhoc" → agent="oracle", suite="adhoc")
# ---------------------------------------------------------------------------


def parse_eval_name(eval_name: str, sep: str = "__") -> tuple[str, str]:
    """Split Harbor eval name into (agent, suite).

    Convention: ``<agent>__<suite>``.  When no separator is present the
    entire string is treated as the suite name with an "unknown" agent.
    """
    if sep in eval_name:
        agent, suite = eval_name.split(sep, 1)
        return agent or "unknown", suite or "adhoc"
    return "unknown", eval_name


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1]
        elif "+" in ts[10:]:
            ts = ts[: ts.index("+", 10)]
        elif ts.count("-") > 2:
            ts = ts[: ts.rfind("-")]
        return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _elapsed_s(start: str | None, end: str | None) -> float:
    a, b = _parse_iso(start), _parse_iso(end)
    if a and b:
        return max(0.0, (b - a).total_seconds())
    return 0.0


def _wall_clock(trial: dict) -> float:
    return _elapsed_s(trial.get("started_at"), trial.get("finished_at"))


def _setup_wall_clock(trial: dict) -> float:
    return _elapsed_s(trial.get("started_at"), trial.get("finished_at"))


def _agent_wall_clock(trial: dict) -> float:
    exe = trial.get("agent_execution") or {}
    return _elapsed_s(exe.get("started_at"), exe.get("finished_at"))


def _verifier_wall_clock(trial: dict) -> float:
    vr = trial.get("verifier") or {}
    return _elapsed_s(vr.get("started_at"), vr.get("finished_at"))


# ---------------------------------------------------------------------------
# Trial → Cell conversion
# ---------------------------------------------------------------------------


def _mean_reward(rewards: dict[str, float]) -> float:
    if not rewards:
        return 0.0
    vals = [v for v in rewards.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def trial_to_cell(trial: dict, agent: str, suite: str) -> dict[str, Any]:
    """Convert a single Harbor trial result.json into a bench-cockpit cell dict."""
    verifier_result = trial.get("verifier_result") or {}
    rewards = verifier_result.get("rewards") or {}
    agent_result = trial.get("agent_result") or {}
    agent_info = trial.get("agent_info") or {}

    ok = bool(rewards)
    reward_val = _mean_reward(rewards)
    wall = _wall_clock(trial)
    agent_wall = _agent_wall_clock(trial)
    verifier_wall = _verifier_wall_clock(trial)

    tokens_in = agent_result.get("n_input_tokens") or 0
    tokens_out = agent_result.get("n_output_tokens") or 0
    tps = (tokens_out / agent_wall) if agent_wall > 0 and tokens_out else 0.0

    trial_name = trial.get("trial_name", trial.get("id", "unknown"))
    task_name = trial.get("task_name", suite)

    has_error = trial.get("exception_info") is not None

    return {
        "suite": suite,
        "task_id": trial_name,
        "task_title": task_name,
        "difficulty": "medium",
        "variant": agent,
        "ok": ok and not has_error,
        "wall_clock_s": round(wall, 3),
        "tokens_per_second": round(tps, 2),
        "first_token_latency_ms": 0.0,
        "peak_rss_mb": 0.0,
        "peak_gpu_mem_mb": 0.0,
        "energy_proxy_joules": 0.0,
        "pass_at_1": round(reward_val, 4),
        "gen_ok": round(reward_val, 4),
        "partial_credit": round(reward_val, 4),
        "judge_score": round(reward_val, 4),
        "intent_preservation_rate": 0.0,
        "hallucination_count": 0,
        "tool_call_success_rate": 1.0,
        "retry_count": 0,
        "format_compliance_rate": round(reward_val, 4),
        "reply": "",
        "reply_full": "",
        "prompt": "",
        "expected_answer": "",
        "scoring_method": "harbor_verifier",
        "total_tokens_in": int(tokens_in or 0),
        "total_tokens_out": int(tokens_out or 0),
        "cost_usd": float(agent_result.get("cost_usd") or 0.0),
        "progress_trace": [],
        "chat_trace": [],
        "failure_analysis": {
            "primary_factor": "ok" if ok and not has_error else "error"
        },
        "assignment": {
            "title": task_name,
            "description": f"Harbor eval `{suite}` via `{agent}`.",
            "acceptance": f"Harbor verifier reward > 0.",
            "rubric": f"Harbor verifier reward > 0.",
        },
        "model_name": agent_info.get("name", agent),
        "model_version": agent_info.get("version", ""),
        "metadata": {
            "source": "harbor",
            "trial_name": trial_name,
            "job_id": trial.get("config", {}).get("job_id", ""),
        },
        "created_at": trial.get("started_at", ""),
        "completed_at": trial.get("finished_at", ""),
    }


# ---------------------------------------------------------------------------
# Job-level aggregation → cells + summary
# ---------------------------------------------------------------------------


def discover_trials(run_dir: Path, job_result: dict) -> list[dict]:
    """Discover and load all trial result.json files under run_dir.

    A Harbor job directory layout:
        <run_dir>/
            result.json              ← job-level
            <trial_name>/
                result.json          ← trial-level

    Returns a list of trial-level dicts that were successfully loaded.
    """
    trials = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        trial_path = child / "result.json"
        if trial_path.exists():
            trial = load_trial_result(trial_path)
            if trial is not None:
                trials.append(trial)
    return trials


def _eval_name_from_job(job_result: dict) -> str:
    """Extract the eval name from the job stats."""
    evals = job_result.get("stats", {}).get("evals", {})
    if len(evals) == 1:
        return next(iter(evals))
    return ""


def convert_job(run_dir: Path) -> dict[str, Any]:
    """Convert a Harbor run directory to bench-cockpit results format.

    Accepts either:
      - A job-level run directory (contains result.json at top level + trial dirs)
      - A parent .runs/ directory (iterates child dirs that look like jobs)

    Returns a dict with ``summary`` and ``cells`` keys.
    """
    job_result_path = run_dir / "result.json"
    if not job_result_path.exists():
        raise FileNotFoundError(f"No result.json in {run_dir}")

    job_result = load_job_result(job_result_path)
    trials = discover_trials(run_dir, job_result)

    # If no per-trial files, try to synthesize from the job-level stats alone
    if not trials:
        trials = _synthesize_trials_from_job(job_result)

    cells: list[dict] = []
    evals = job_result.get("stats", {}).get("evals", {})

    for trial in trials:
        # Try to determine agent and suite from trial metadata first
        agent = trial.get("agent_info", {}).get("name") or "unknown"
        suite = trial.get("task_name", "unknown")

        # Fallback to parsing the eval name from job stats
        if suite == "unknown" and evals:
            eval_name = _eval_name_from_job(job_result)
            if eval_name:
                agent, suite = parse_eval_name(eval_name)

        cells.append(trial_to_cell(trial, agent, suite))

    if not cells:
        raise ValueError(f"No cells produced from {run_dir}")

    return _build_cockpit_output(cells, job_result)


def _synthesize_trials_from_job(job_result: dict) -> list[dict]:
    """Build minimal trial dicts from job-level stats when per-trial files are absent."""
    evals = job_result.get("stats", {}).get("evals", {})
    trials = []
    for eval_name, eval_stats in evals.items():
        agent, suite = parse_eval_name(eval_name)
        n_trials = eval_stats.get("n_trials", 0)
        metrics = eval_stats.get("metrics", [])
        mean_val = metrics[0].get("mean", 0.0) if metrics else 0.0

        for i in range(n_trials):
            trials.append(
                {
                    "id": f"{eval_name}__trial_{i}",
                    "trial_name": f"{eval_name}__trial_{i}",
                    "task_name": f"{agent}/{suite}",
                    "agent_info": {"name": agent, "version": ""},
                    "agent_result": {
                        "n_input_tokens": job_result.get("stats", {}).get(
                            "n_input_tokens"
                        ),
                        "n_output_tokens": job_result.get("stats", {}).get(
                            "n_output_tokens"
                        ),
                        "cost_usd": job_result.get("stats", {}).get("cost_usd"),
                    },
                    "verifier_result": {"rewards": {"reward": mean_val}},
                    "started_at": job_result.get("started_at"),
                    "finished_at": job_result.get("finished_at"),
                    "exception_info": None,
                    "config": {"job_id": job_result.get("id", "")},
                }
            )
    return trials


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


def _build_cockpit_output(cells: list[dict], job_result: dict) -> dict[str, Any]:
    """Build the full bench-cockpit {summary, cells} structure."""
    variants: dict[str, dict] = {}
    difficulty_mix: dict[str, int] = {}
    suites_seen: set[str] = set()

    for cell in cells:
        v = cell["variant"]
        s = cell["suite"]
        d = cell.get("difficulty", "unknown")
        suites_seen.add(s)
        difficulty_mix[d] = difficulty_mix.get(d, 0) + 1

        if v not in variants:
            variants[v] = {
                "n_cells": 0,
                "pass_at_1_sum": 0.0,
                "wall_sum": 0.0,
                "partial_sum": 0.0,
                "format_sum": 0.0,
                "ok_count": 0,
                "hall_sum": 0,
            }
        va = variants[v]
        va["n_cells"] += 1
        va["pass_at_1_sum"] += cell.get("pass_at_1", 0.0)
        va["wall_sum"] += cell.get("wall_clock_s", 0.0)
        va["partial_sum"] += cell.get("partial_credit", 0.0)
        va["format_sum"] += cell.get("format_compliance_rate", 0.0)
        if cell.get("ok"):
            va["ok_count"] += 1
        va["hall_sum"] += cell.get("hallucination_count", 0)

    by_variant: dict[str, dict] = {}
    for v, va in variants.items():
        n = va["n_cells"] or 1
        by_variant[v] = {
            "n_cells": va["n_cells"],
            "pass_at_1": round(va["pass_at_1_sum"] / n, 4),
            "mean_wall_clock_s": round(va["wall_sum"] / n, 3),
            "mean_partial_credit": round(va["partial_sum"] / n, 4),
            "mean_format_compliance": round(va["format_sum"] / n, 4),
            "mean_intent_preservation": 0.0,
            "n_hallucinations": va["hall_sum"],
            "ok_count": va["ok_count"],
        }

    stats = job_result.get("stats", {})
    model = _eval_name_from_job(job_result)
    if model:
        agent, _ = parse_eval_name(model)
        model = agent

    return {
        "summary": {
            "meta": {
                "model": model or "unknown",
                "mlx_url": "",
                "n_suites": len(suites_seen),
                "n_tasks_per_suite": len(cells) // max(len(suites_seen), 1),
                "variants": sorted(by_variant.keys()),
                "n_cells": len(cells),
                "difficulty_mix": difficulty_mix,
            },
            "by_variant": by_variant,
        },
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Directory scanning (convert multiple jobs in one pass)
# ---------------------------------------------------------------------------


def convert_runs_root(root: Path) -> dict[str, Any]:
    """Scan a .runs/ directory and convert all Harbor jobs found."""
    all_cells: list[dict] = []
    job_count = 0

    for job_dir in sorted(root.iterdir()):
        if not job_dir.is_dir():
            continue
        job_result_path = job_dir / "result.json"
        if not job_result_path.exists():
            continue
        try:
            result = convert_job(job_dir)
            all_cells.extend(result["cells"])
            job_count += 1
        except (ValueError, FileNotFoundError) as exc:
            print(f"warn: skip {job_dir.name}: {exc}", file=sys.stderr)

    if not all_cells:
        raise ValueError(f"No cells produced from {root}")

    return _build_cockpit_output(
        all_cells,
        {
            "id": f"batch-{root.name}",
            "stats": {"evals": {}},
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Harbor result.json → bench-cockpit cells format.",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to a Harbor job directory (contains result.json + trial dirs).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: stdout).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Treat run_dir as .runs/ root and convert all jobs found.",
    )
    args = parser.parse_args(argv)

    if not args.run_dir.is_dir():
        print(f"error: {args.run_dir} is not a directory", file=sys.stderr)
        return 1

    try:
        if args.all:
            result = convert_runs_root(args.run_dir)
        else:
            result = convert_job(args.run_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output = json.dumps(result, indent=2, default=str) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(f"wrote {args.output} ({len(result['cells'])} cells)", file=sys.stderr)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
