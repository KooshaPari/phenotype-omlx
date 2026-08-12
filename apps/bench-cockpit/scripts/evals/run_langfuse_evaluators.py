#!/usr/bin/env python3
"""Seed Langfuse traces/generations from V5 JSON and run Minimax judges.

Canonical observability backend (OSS / self-hostable). LangSmith removed.

  OBSERVABILITY_BACKEND=langfuse
  LANGFUSE_PUBLIC_KEY=pk-...
  LANGFUSE_SECRET_KEY=sk-...
  LANGFUSE_BASE_URL=https://us.cloud.langfuse.com

  # Hosted judges need Settings → LLM Connections → Minimax (anthropic adapter)
  python3 scripts/evals/setup_langfuse_judges.py
  python3 scripts/evals/run_langfuse_evaluators.py sync|seed|judge --limit 40
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import stat
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# Allow-list for dotenv-loaded keys. Token-bearing env vars
# (OPENAI_API_KEY / ANTHROPIC_API_KEY / MLX_SERVER_URL / etc.) must NOT
# be inherited from a local .env into the harbor subprocess unless the
# user explicitly exports them. Keep this list short and additive.
_DOTENV_ALLOWED_PREFIXES = ("PORTAGE_", "LANGFUSE_", "HARBOR_LANGFUSE_", "OBSERVABILITY_BACKEND")
_DOTENV_PERMISSIVE_MASK = 0o077  # refuse .env that is group/other writable


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.is_file():
        return
    try:
        mode = stat.S_IMODE(env.stat().st_mode)
    except OSError:
        return
    if mode & _DOTENV_PERMISSIVE_MASK:
        warnings.warn(
            f"dotenv: refusing {env} with permissive mode {oct(mode)} "
            f"(must not be group/other writable)",
            stacklevel=2,
        )
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if not k or not k.startswith(_DOTENV_ALLOWED_PREFIXES):
            continue
        if not v:
            warnings.warn(
                f"dotenv: empty token for {k} in {env} (non-fatal)", stacklevel=2
            )
        if k not in os.environ:
            os.environ[k] = v


_load_dotenv()

BASE = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or "https://cloud.langfuse.com"
BASE = BASE.rstrip("/")
PUB = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
SEC = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
MINIMAX_ENDPOINT = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_MODEL = os.environ.get("MINIMAX_JUDGE_MODEL", "MiniMax-M3")

LLM_RUBRICS = [
    (
        "correctness",
        "Score 1 only if the reply correctly solves the task; 0 if it echoes the prompt, "
        "repeats blocks, or fails the task. Partial credit 0.3-0.7 for partial solutions.",
    ),
    (
        "hallucination",
        "Score 1 if the reply does not invent APIs/paths/facts; 0 if it fabricates. "
        "Echoing instructions is hallucination of competence — score low.",
    ),
    (
        "code_checker",
        "If the task needs code/bash/diff, score whether the code would run/solve it. "
        "Repeated identical snippets score 0. Non-code tasks: score 0.5.",
    ),
]


def die(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def auth_header() -> str:
    if not PUB or not SEC:
        die("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY required")
    tok = base64.b64encode(f"{PUB}:{SEC}".encode()).decode()
    return "Basic " + tok


def lf_request(method: str, path: str, body: Any | None = None) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": auth_header(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode()
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            parsed: Any = json.loads(raw) if raw else {"error": raw}
        except json.JSONDecodeError:
            parsed = {"error": raw}
        return e.code, parsed


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def default_data_path() -> Path:
    """Resolve V5 cells JSON without inventing new trial runs.

    Prefer ``BENCH_DATA``, then the historical short V5 under
    ``repos/pheno-harness/...`` (read-only; no stock_vs_ours re-run), then
    cockpit smoke fixtures.
    """
    env = os.environ.get("BENCH_DATA")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in here.parents:
        if parent.name == "repos":
            candidates.append(
                parent
                / "pheno-harness"
                / "bench"
                / "results"
                / "stock-vs-ours"
                / "run-v5-qwen35-08b.json"
            )
            break
    candidates.append(
        Path(
            "/Users/kooshapari/CodeProjects/Phenotype/repos/pheno-harness/bench/results/"
            "stock-vs-ours/run-v5-qwen35-08b.json"
        )
    )
    for native in candidates:
        if native.is_file():
            return native
    return ROOT / "fixtures" / "smoke_results.json"


def parse_judge_payload(text: str) -> tuple[float, str]:
    """Extract ``score`` / ``reason`` from a judge model response body."""
    m = re.search(r"\{[^{}]+\}", text, re.S)
    if not m:
        return 0.0, f"unparseable:{text[:120]}"
    try:
        parsed = json.loads(m.group(0))
        return float(parsed.get("score", 0)), str(parsed.get("reason", ""))[:240]
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0, f"bad_json:{text[:120]}"


def load_cells(path: Path, limit: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    cells = data.get("cells") or []
    return cells[:limit]


def seed(limit: int, data_path: Path) -> dict[str, Any]:
    cells = load_cells(data_path, limit)
    ts = now_iso()
    batch: list[dict[str, Any]] = []
    traces: list[str] = []
    for c in cells:
        tid = str(uuid.uuid4())
        oid = str(uuid.uuid4())
        traces.append(tid)
        gen_ok = float(c.get("gen_ok") if c.get("gen_ok") is not None else c.get("pass_at_1") or 0)
        prompt = (c.get("prompt") or "")[:2000]
        reply = (c.get("reply") or "")[:2000]
        inp = {
            "prompt": prompt,
            "suite": c.get("suite"),
            "task_id": c.get("task_id"),
            "variant": c.get("variant"),
        }
        out = {
            "reply": reply,
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
                    "tags": ["bench-cockpit", str(c.get("suite")), str(c.get("variant"))],
                    "metadata": {
                        "suite": c.get("suite"),
                        "task_id": c.get("task_id"),
                        "variant": c.get("variant"),
                        "gen_ok": gen_ok,
                        "verified_pass_at_1": c.get("verified_pass_at_1", 0),
                        "source": "bench-cockpit",
                    },
                    "input": inp,
                    "output": out,
                },
            }
        )
        # GENERATION observations so hosted observation-target eval rules fire.
        batch.append(
            {
                "id": str(uuid.uuid4()),
                "type": "generation-create",
                "timestamp": ts,
                "body": {
                    "id": oid,
                    "traceId": tid,
                    "name": "bench-cell",
                    "model": str(c.get("model") or c.get("variant") or "bench"),
                    "input": inp,
                    "output": out,
                    "metadata": {
                        "suite": c.get("suite"),
                        "task_id": c.get("task_id"),
                        "variant": c.get("variant"),
                        "source": "bench-cockpit",
                    },
                    "startTime": ts,
                    "endTime": ts,
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
                    "comment": "generation success (not verified pass@1)",
                },
            }
        )
    code, body = lf_request("POST", "/api/public/ingestion", {"batch": batch})
    return {
        "status_code": code,
        "cells": len(cells),
        "events": len(batch),
        "trace_ids": traces,
        "ingestion": body,
        "dashboard": BASE,
    }


def _minimax_key() -> str:
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if key:
        return key
    try:
        out = subprocess.check_output(
            ["security", "find-generic-password", "-s", "minimax-coding-plan", "-w"],
            text=True,
        ).strip()
        if out:
            os.environ["MINIMAX_API_KEY"] = out
            return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return ""


def minimax_judge(prompt: str, reply: str, rubric: str) -> tuple[float, str]:
    api_key = _minimax_key()
    if not api_key:
        return 0.0, "no_minimax_key"
    body = json.dumps(
        {
            "model": MINIMAX_MODEL,
            "max_tokens": 256,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "You are a strict evaluation judge for LLM agent outputs.\n"
                        f"Rubric: {rubric}\n"
                        'Return ONLY JSON: {"score": 0.0-1.0, "reason": "short"}\n\n'
                        f"INPUT: {prompt[:1500]}\nOUTPUT: {reply[:2000]}\n"
                    ),
                }
            ],
        }
    ).encode()
    req = urllib.request.Request(
        MINIMAX_ENDPOINT,
        data=body,
        method="POST",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return 0.0, f"judge_error:{e}"
    text = ""
    for block in data.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text += block.get("text") or ""
    return parse_judge_payload(text)


def judge(limit: int, trace_ids: list[str] | None = None) -> dict[str, Any]:
    if not _minimax_key():
        die(
            "MINIMAX_API_KEY required for live LLM-as-judge "
            "(env or keychain service minimax-coding-plan); refusing silent zero scores"
        )
    traces: list[dict[str, Any]] = []
    if trace_ids:
        for tid in trace_ids[:limit]:
            code, body = lf_request("GET", f"/api/public/traces/{tid}")
            if code >= 300:
                die(f"trace {tid} {code}: {body}")
            # Some APIs wrap under "data"
            tr = body.get("data") if isinstance(body.get("data"), dict) else body
            if isinstance(tr, dict) and tr.get("id"):
                traces.append(tr)
    else:
        code, body = lf_request("GET", f"/api/public/traces?limit={limit}")
        if code >= 300:
            die(f"traces {code}: {body}")
        traces = body.get("data") or body.get("traces") or []
    ts = now_iso()
    batch: list[dict[str, Any]] = []
    scored = 0
    correctness_by_trace: dict[str, float] = {}
    for tr in traces:
        tid = tr.get("id")
        if not tid:
            continue
        inp = tr.get("input") or {}
        out = tr.get("output") or {}
        prompt = str(inp.get("prompt") or json.dumps(inp)[:1500])
        reply = str(out.get("reply") or json.dumps(out)[:2000])
        for key, rubric in LLM_RUBRICS:
            score, reason = minimax_judge(prompt, reply, rubric)
            if reason.startswith("no_minimax_key") or reason.startswith("judge_error:"):
                die(f"judge failed for trace {tid} / {key}: {reason}")
            batch.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": "score-create",
                    "timestamp": ts,
                    "body": {
                        "id": str(uuid.uuid4()),
                        "traceId": tid,
                        "name": key,
                        "value": score,
                        "dataType": "NUMERIC",
                        "comment": reason,
                    },
                }
            )
            scored += 1
            if key == "correctness":
                correctness_by_trace[tid] = score
    if not batch:
        return {"traces": len(traces), "scores": 0, "note": "no traces to judge"}
    code, ing = lf_request("POST", "/api/public/ingestion", {"batch": batch})
    return {
        "status_code": code,
        "traces": len(traces),
        "scores": scored,
        "ingestion": ing,
        "llm_enabled": True,
        "judge_score_by_trace": correctness_by_trace,
        "mean_judge_score": (
            round(sum(correctness_by_trace.values()) / len(correctness_by_trace), 4)
            if correctness_by_trace
            else None
        ),
    }


def run_all(limit: int, data_path: Path) -> dict[str, Any]:
    """Seed historical/fixture cells then live-judge those seeded traces only."""
    seeded = seed(limit, data_path)
    ids = list(seeded.get("trace_ids") or [])
    judged = judge(limit, trace_ids=ids)
    return {"seed": seeded, "judge": judged}


def sync_hosted() -> dict[str, Any]:
    """Delegate to setup_langfuse_judges.py (hosted Minimax evaluators + rules)."""
    script = ROOT / "scripts" / "evals" / "setup_langfuse_judges.py"
    if not script.is_file():
        die(f"missing {script}")
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out: dict[str, Any] = {"returncode": proc.returncode}
    try:
        out["result"] = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except json.JSONDecodeError:
        out["stdout"] = proc.stdout[-4000:]
    if proc.stderr:
        out["stderr"] = proc.stderr[-2000:]
    if proc.returncode != 0:
        die(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "action",
        choices=["seed", "judge", "all", "status", "sync"],
        help="all = seed historical/fixture cells then live-judge traces",
    )
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--data", type=Path, default=None)
    args = ap.parse_args()
    if args.action == "status":
        code, health = lf_request("GET", "/api/public/health")
        _, projects = lf_request("GET", "/api/public/projects")
        _, conns = lf_request("GET", "/api/public/llm-connections?limit=20")
        print(
            json.dumps(
                {
                    "health_code": code,
                    "health": health,
                    "projects": projects,
                    "llm_connections": conns,
                    "base": BASE,
                    "default_data": str(default_data_path()),
                    "default_data_exists": default_data_path().is_file(),
                },
                indent=2,
            )
        )
        return
    if args.action == "sync":
        print(json.dumps(sync_hosted(), indent=2))
        return
    data_path = args.data or default_data_path()
    if args.action == "seed":
        out = seed(args.limit, data_path)
        print(json.dumps(out, indent=2))
        return
    if args.action == "judge":
        out = judge(args.limit)
        print(json.dumps(out, indent=2))
        return
    if args.action == "all":
        out = run_all(args.limit, data_path)
        print(json.dumps(out, indent=2))
        return


if __name__ == "__main__":
    main()
