#!/usr/bin/env python3
"""Seed Langfuse traces from V5 JSON and optionally run Minimax offline judges.

Preferred observability backend (OSS / self-hostable). LangSmith remains optional.

  OBSERVABILITY_BACKEND=langfuse
  LANGFUSE_PUBLIC_KEY=pk-...
  LANGFUSE_SECRET_KEY=sk-...
  LANGFUSE_BASE_URL=https://us.cloud.langfuse.com

  python3 scripts/evals/run_langfuse_evaluators.py seed --limit 40
  python3 scripts/evals/run_langfuse_evaluators.py judge --limit 20
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.is_file():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
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
    env = os.environ.get("BENCH_DATA")
    if env:
        return Path(env)
    native = Path(
        "/Users/kooshapari/CodeProjects/Phenotype/pheno-harness/bench/results/"
        "stock-vs-ours/run-v5-qwen35-08b.json"
    )
    if native.is_file():
        return native
    return ROOT / "fixtures" / "smoke_results.json"


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
        traces.append(tid)
        gen_ok = float(c.get("gen_ok") if c.get("gen_ok") is not None else c.get("pass_at_1") or 0)
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
                    "input": {
                        "prompt": (c.get("prompt") or "")[:2000],
                        "suite": c.get("suite"),
                        "task_id": c.get("task_id"),
                        "variant": c.get("variant"),
                    },
                    "output": {
                        "reply": (c.get("reply") or "")[:2000],
                        "gen_ok": gen_ok,
                        "partial_credit": c.get("partial_credit"),
                        "pass_at_1": c.get("pass_at_1"),
                        "wall_clock_s": c.get("wall_clock_s"),
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
    m = re.search(r"\{[^{}]+\}", text, re.S)
    if not m:
        return 0.0, f"unparseable:{text[:120]}"
    try:
        parsed = json.loads(m.group(0))
        return float(parsed.get("score", 0)), str(parsed.get("reason", ""))[:240]
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0, f"bad_json:{text[:120]}"


def judge(limit: int) -> dict[str, Any]:
    code, body = lf_request("GET", f"/api/public/traces?limit={limit}")
    if code >= 300:
        die(f"traces {code}: {body}")
    traces = body.get("data") or body.get("traces") or []
    ts = now_iso()
    batch: list[dict[str, Any]] = []
    scored = 0
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
    if not batch:
        return {"traces": len(traces), "scores": 0, "note": "no traces to judge"}
    code, ing = lf_request("POST", "/api/public/ingestion", {"batch": batch})
    return {
        "status_code": code,
        "traces": len(traces),
        "scores": scored,
        "ingestion": ing,
        "llm_enabled": bool(_minimax_key()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["seed", "judge", "status"])
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--data", type=Path, default=None)
    args = ap.parse_args()
    if args.action == "status":
        code, health = lf_request("GET", "/api/public/health")
        _, projects = lf_request("GET", "/api/public/projects")
        print(json.dumps({"health_code": code, "health": health, "projects": projects, "base": BASE}, indent=2))
        return
    if args.action == "seed":
        out = seed(args.limit, args.data or default_data_path())
        print(json.dumps(out, indent=2))
        return
    if args.action == "judge":
        out = judge(args.limit)
        print(json.dumps(out, indent=2))
        return


if __name__ == "__main__":
    main()
