#!/usr/bin/env python3
"""Convert one Harbor trial result.json to V5-style {summary, cells} for Langfuse seed.

Cockpit `/api/langfuse/setup` and `run_langfuse_evaluators.py seed` expect cells JSON
(not Harbor's trial schema). Harbor's native upload path is LangSmith.

  python harbor_result_to_cells.py TRIAL/result.json -o cells.json
  python harbor_result_to_cells.py TRIAL/result.json --seed   # Langfuse ingestion
  # Or load cells as BENCH_DATA then:
  #   curl -s -X POST http://127.0.0.1:8090/api/langfuse/setup -H 'Content-Type: application/json' -d '{"max_cells":40}'

Does not print secrets. Loads keys from apps/bench-cockpit/.env when --seed.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]  # phenotype-omlx
COCKPIT_ENV = REPO / "apps" / "bench-cockpit" / ".env"


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _wall_clock_s(trial: dict[str, Any]) -> float:
    start = trial.get("started_at")
    end = trial.get("finished_at")
    if not start or not end:
        return 0.0
    try:
        a = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        return max(0.0, (b - a).total_seconds())
    except ValueError:
        return 0.0


def _suite_and_task(trial: dict[str, Any]) -> tuple[str, str]:
    name = str(trial.get("task_name") or "")
    if "/" in name:
        suite, task = name.split("/", 1)
        return suite or "harbor", task or "unknown"
    tid = trial.get("task_id")
    if isinstance(tid, dict):
        if tid.get("name"):
            return str(tid.get("org") or "harbor"), str(tid["name"])
        path = tid.get("path")
        if path:
            parts = Path(str(path)).parts
            return "harbor", parts[-1] if parts else (name or "unknown")
    return "harbor", name or "unknown"


def _reward(trial: dict[str, Any]) -> float:
    vr = trial.get("verifier_result") or {}
    rewards = vr.get("rewards") or {}
    if "reward" in rewards:
        try:
            return float(rewards["reward"])
        except (TypeError, ValueError):
            pass
    return 0.0


def trial_to_cell(trial: dict[str, Any], variant: str | None = None) -> dict[str, Any]:
    suite, task_id = _suite_and_task(trial)
    agent = (trial.get("agent_info") or {}) or {}
    cfg_agent = ((trial.get("config") or {}).get("agent") or {}) or {}
    model = agent.get("model_info") or cfg_agent.get("model_name") or agent.get("name") or "oracle"
    arm = variant or str(cfg_agent.get("name") or agent.get("name") or "oracle")
    reward = _reward(trial)
    ok = reward >= 0.999 and trial.get("exception_info") is None
    wall = _wall_clock_s(trial)
    prompt = f"harbor task: {trial.get('task_name') or task_id}"
    reply = f"reward={reward}"
    if trial.get("exception_info"):
        reply = f"exception: {trial.get('exception_info')}"
    ar = trial.get("agent_result") or {}
    return {
        "suite": suite,
        "task_id": task_id,
        "task_title": str(trial.get("task_name") or task_id),
        "difficulty": "unknown",
        "variant": arm,
        "ok": ok,
        "wall_clock_s": wall,
        "tokens_per_second": 0.0,
        "pass_at_1": reward,
        "gen_ok": reward,
        "verified_pass_at_1": reward if ok else 0.0,
        "partial_credit": reward,
        "judge_score": reward,
        "reply": reply[:2000],
        "prompt": prompt[:2000],
        "scoring_method": "harbor_verifier_reward",
        "model_name": str(model),
        "created_at": trial.get("started_at") or "",
        "completed_at": trial.get("finished_at") or "",
        "total_tokens_in": int(ar.get("n_input_tokens") or 0),
        "total_tokens_out": int(ar.get("n_output_tokens") or 0),
        "cost_usd": float(ar.get("cost_usd") or 0.0),
        "error_message": "" if ok else str(trial.get("exception_info") or "fail"),
        "error_code": "" if ok else "harbor_fail",
        "metadata": {
            "source": "harbor_result",
            "trial_name": str(trial.get("trial_name") or ""),
            "trial_id": str(trial.get("id") or ""),
            "task_checksum": str(trial.get("task_checksum") or ""),
            "evidence_label": "harbor_verified",
        },
    }


def load_trial(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Job-level result.json has n_total_trials - reject; need trial result.
    if "n_total_trials" in data and "verifier_result" not in data:
        raise SystemExit(
            f"error: {path} looks like a Harbor job result, not a trial result.json "
            "(need .../<trial_name>/result.json with verifier_result)"
        )
    if "verifier_result" not in data and "task_name" not in data:
        raise SystemExit(f"error: {path} is not a Harbor trial result.json")
    return data


def wrap_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    variants = sorted({c["variant"] for c in cells})
    return {
        "summary": {
            "meta": {
                "model": cells[0].get("model_name", "harbor") if cells else "harbor",
                "n_suites": len({c["suite"] for c in cells}),
                "n_cells": len(cells),
                "variants": variants,
                "source": "harbor_result_to_cells",
            },
            "by_variant": {},
        },
        "cells": cells,
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def seed_langfuse(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Same shape as cockpit /api/langfuse/setup + run_langfuse_evaluators seed."""
    _load_dotenv(COCKPIT_ENV)
    base = (
        os.environ.get("LANGFUSE_BASE_URL")
        or os.environ.get("LANGFUSE_HOST")
        or "https://cloud.langfuse.com"
    ).rstrip("/")
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    sec = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    if not pub or not sec:
        raise SystemExit("error: LANGFUSE_PUBLIC_KEY/SECRET_KEY missing (cockpit .env)")
    tok = base64.b64encode(f"{pub}:{sec}".encode()).decode()
    ts = now_iso()
    batch: list[dict[str, Any]] = []
    traces: list[str] = []
    for c in cells:
        tid = str(uuid.uuid4())
        oid = str(uuid.uuid4())
        traces.append(tid)
        gen_ok = float(c.get("gen_ok") if c.get("gen_ok") is not None else c.get("pass_at_1") or 0)
        inp = {
            "prompt": (c.get("prompt") or "")[:2000],
            "suite": c.get("suite"),
            "task_id": c.get("task_id"),
            "variant": c.get("variant"),
        }
        out = {
            "reply": (c.get("reply") or "")[:2000],
            "ok": c.get("ok"),
            "gen_ok": gen_ok,
            "partial_credit": c.get("partial_credit"),
            "pass_at_1": c.get("pass_at_1"),
            "wall_clock_s": c.get("wall_clock_s"),
        }
        batch.append(
            {
                "id": str(uuid.uuid4()),
                "type": "trace-create",
                "timestamp": ts,
                "body": {
                    "id": tid,
                    "name": f"{c.get('suite')}/{c.get('task_id')}/{c.get('variant')}",
                    "tags": ["harbor", "bench-cockpit", str(c.get("suite")), str(c.get("variant"))],
                    "metadata": {
                        "suite": c.get("suite"),
                        "task_id": c.get("task_id"),
                        "variant": c.get("variant"),
                        "gen_ok": gen_ok,
                        "verified_pass_at_1": c.get("verified_pass_at_1", 0),
                        "pass_at_1": c.get("pass_at_1"),
                        "source": "harbor_result_to_cells",
                    },
                    "input": inp,
                    "output": out,
                },
            }
        )
        batch.append(
            {
                "id": str(uuid.uuid4()),
                "type": "generation-create",
                "timestamp": ts,
                "body": {
                    "id": oid,
                    "traceId": tid,
                    "name": "harbor-cell",
                    "model": str(c.get("model_name") or c.get("variant") or "harbor"),
                    "input": inp,
                    "output": out,
                    "startTime": ts,
                    "endTime": ts,
                    "metadata": {
                        "suite": c.get("suite"),
                        "task_id": c.get("task_id"),
                        "variant": c.get("variant"),
                        "source": "harbor_result_to_cells",
                    },
                },
            }
        )
        batch.append(
            {
                "id": str(uuid.uuid4()),
                "type": "score-create",
                "timestamp": ts,
                "body": {
                    "id": str(uuid.uuid4()),
                    "traceId": tid,
                    "observationId": oid,
                    "name": "gen_ok",
                    "value": gen_ok,
                    "dataType": "NUMERIC",
                    "comment": "harbor verifier reward",
                },
            }
        )
    body = json.dumps({"batch": batch}).encode()
    req = urllib.request.Request(
        f"{base}/api/public/ingestion",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Basic {tok}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
            code = resp.status
            parsed: Any = json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        code = e.code
        try:
            parsed = json.loads(raw) if raw else {"error": raw}
        except json.JSONDecodeError:
            parsed = {"error": raw[:500]}
    return {
        "status_code": code,
        "cells_seeded": len(cells),
        "events": len(batch),
        "trace_ids": traces,
        "ingestion_ok": code < 300,
        "dashboard_url": base,
        # omit raw ingestion body keys that might echo secrets - keep status only
        "ingestion_errors": (parsed.get("errors") if isinstance(parsed, dict) else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result_json", type=Path, help="Harbor trial .../<trial>/result.json")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Write cells JSON")
    ap.add_argument("--variant", default=None, help="Override cell variant (default: agent name)")
    ap.add_argument("--seed", action="store_true", help="POST Langfuse ingestion (keys from cockpit .env)")
    ap.add_argument("--dry-run", action="store_true", help="Print summary only (default if no -o/--seed)")
    args = ap.parse_args()

    trial = load_trial(args.result_json)
    cell = trial_to_cell(trial, variant=args.variant)
    payload = wrap_cells([cell])

    print(
        json.dumps(
            {
                "input": str(args.result_json),
                "cell_count": 1,
                "suite": cell["suite"],
                "task_id": cell["task_id"],
                "variant": cell["variant"],
                "pass_at_1": cell["pass_at_1"],
                "ok": cell["ok"],
            },
            indent=2,
        )
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)

    if args.seed:
        result = seed_langfuse(payload["cells"])
        print(json.dumps(result, indent=2))
        if not result.get("ingestion_ok"):
            raise SystemExit(1)
    elif not args.output and not args.dry_run:
        # default dry-run already printed summary
        pass


if __name__ == "__main__":
    main()
