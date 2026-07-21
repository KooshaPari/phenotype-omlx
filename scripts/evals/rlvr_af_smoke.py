#!/usr/bin/env python3
"""L6 — RLVR-AF Phase-1 smoke: hard verifier on micro fixtures (no LoRA, no MLX).

Wraps pheno-harness ``bench.rlvr_af`` HeuristicVerifier over a tiny exact-match
corpus. Produces a Harbor/LangSmith-friendly envelope under
``research/rlvr_af/``.

Requires pheno-harness on disk (default path below) — fail loud if missing.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PHENO_DEFAULT = Path("/Users/kooshapari/CodeProjects/Phenotype/pheno-harness")

# Micro verifiable tasks — hard string match (RLVR-AF Layer 2A).
MICRO_TASKS = [
    {
        "task_id": "schema-json-ok",
        "prompt": "Return JSON {\"ok\": true}",
        "completion": '{"ok": true}',
        "expected": '"ok": true',
    },
    {
        "task_id": "needle-exact",
        "prompt": "Secret code?",
        "completion": "The secret code is 42-alpha.",
        "expected": "42-alpha",
    },
    {
        "task_id": "empty-fail",
        "prompt": "Say hi",
        "completion": "",
        "expected": "hi",
    },
    {
        "task_id": "error-prefix-fail",
        "prompt": "Compute 1+1",
        "completion": "error: backend down",
        "expected": "2",
    },
    {
        "task_id": "partial-credit",
        "prompt": "List primary colors",
        "completion": "Red and blue are important colors in the palette.",
        "expected": "green",
    },
]


def _load_rlvr(pheno: Path):
    if not pheno.is_dir():
        raise SystemExit(f"error: pheno-harness missing at {pheno}")
    sys.path.insert(0, str(pheno))
    from bench.rlvr_af.trace import Artifact, Transition
    from bench.rlvr_af.verify import HeuristicVerifier

    return Artifact, Transition, HeuristicVerifier


def run_smoke(pheno: Path = PHENO_DEFAULT) -> dict:
    Artifact, Transition, HeuristicVerifier = _load_rlvr(pheno)
    verifier = HeuristicVerifier()
    rows = []
    for spec in MICRO_TASKS:
        art = Artifact(
            elapsed_s=0.0,
            prompt=spec["prompt"],
            completion=spec["completion"],
            expected=spec["expected"],
        )
        t = Transition(task_id=spec["task_id"], artifact=art)
        v = verifier.verify(t)
        rows.append(
            {
                "task_id": spec["task_id"],
                "passed": v.passed,
                "reward": v.reward,
                "reason": v.reason,
            }
        )

    rewards = [r["reward"] for r in rows]
    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    n_pass = sum(1 for r in rows if r["passed"])

    # Contract: empty + error must fail; needle + schema must pass
    by_id = {r["task_id"]: r for r in rows}
    contract_ok = (
        by_id["needle-exact"]["passed"]
        and by_id["schema-json-ok"]["passed"]
        and not by_id["empty-fail"]["passed"]
        and not by_id["error-prefix-fail"]["passed"]
    )

    return {
        "experiment": "L6-rlvr-af-smoke",
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": "fixture-hard-verifier",  # no LLM weights
        "pheno_harness": str(pheno),
        "n_tasks": len(rows),
        "n_passed": n_pass,
        "mean_reward": round(mean_r, 4),
        "rows": rows,
        "contract_ok": contract_ok,
        "verdict": "PASS" if contract_ok else "FAIL",
        "notes": [
            "Phase-1 eval-first: hard verifier only (no LoRA / soft judge).",
            "Soft-judge calibration blocked until live non-synthetic L1/L4.",
            "Operator path remains Portage/Harbor + LangSmith for agent trials.",
        ],
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "research" / "rlvr_af"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_smoke()
    (out_dir / "qwen35-08b-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    md = (
        f"# L6 RLVR-AF smoke\n\n"
        f"**Verdict:** `{result['verdict']}` · mean_reward={result['mean_reward']} "
        f"· passed={result['n_passed']}/{result['n_tasks']}\n\n"
        f"| task | passed | reward | reason |\n|------|--------|--------|--------|\n"
        + "\n".join(
            f"| {r['task_id']} | {r['passed']} | {r['reward']} | {r['reason']} |"
            for r in result["rows"]
        )
        + "\n"
    )
    (out_dir / "qwen35-08b-smoke.md").write_text(md)
    # Mirror into pheno-harness when possible
    mirror = PHENO_DEFAULT / "bench" / "results" / "rlvr_af"
    try:
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / "qwen35-08b-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        (mirror / "qwen35-08b-smoke.md").write_text(md)
        print(f"mirrored {mirror}")
    except OSError as e:
        print(f"skip pheno mirror: {e}")
    print(f"wrote {out_dir}/qwen35-08b-smoke.{{json,md}}")
    print("verdict=", result["verdict"])
    return 0 if result["contract_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
