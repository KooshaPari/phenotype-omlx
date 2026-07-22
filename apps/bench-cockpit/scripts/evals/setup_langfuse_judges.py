#!/usr/bin/env python3
"""Register Langfuse hosted LLM-as-judge evaluators against the Minimax connection.

Requires a Langfuse LLM connection (Settings → LLM Connections):
  Provider: Minimax
  Adapter: anthropic
  Base URL: https://api.minimax.io/anthropic
  Custom model: Minimax-M3

Creates score configs, project evaluators, observation (+ optional experiment) rules.
Minimax structured-output preflight is flaky — retries with create-disabled + enable.

  python3 scripts/evals/setup_langfuse_judges.py
  python3 scripts/evals/setup_langfuse_judges.py --status
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

PROVIDER = os.environ.get("LANGFUSE_JUDGE_PROVIDER", "Minimax")
MODEL = os.environ.get("LANGFUSE_JUDGE_MODEL", "Minimax-M3")
DATASET_NAME = "bench-cockpit-v5-cells"

SCORE_CONFIGS = [
    "gen_ok",
    "partial_credit",
    "correctness",
    "hallucination",
    "code_checker",
]

OUTDEF: dict[str, Any] = {
    "dataType": "NUMERIC",
    "score": {
        "description": (
            "Score between 0 and 1. Score 0 if false or negative and 1 if true or positive"
        )
    },
    "reasoning": {"description": "One sentence reasoning for the score"},
}

RUBRICS: list[tuple[str, str]] = [
    (
        "bench-correctness",
        "Evaluate correctness of the generation. Score 1 if it solves the task; "
        "0 if it echoes the prompt, repeats blocks, or fails. Partial 0.3-0.7 for "
        "partial solutions.\nInput: {{input}}\nOutput: {{output}}",
    ),
    (
        "bench-hallucination",
        "Evaluate hallucination risk. Score 1 if reply invents nothing "
        "(no fake APIs/paths/facts); score 0 if it fabricates. Echoing the prompt "
        "counts as low quality.\nInput: {{input}}\nOutput: {{output}}",
    ),
    (
        "bench-code-checker",
        "If the task needs code/bash/diff, score whether the code would run/solve it. "
        "Repeated identical snippets score 0. Non-code tasks: score 0.5.\n"
        "Input: {{input}}\nOutput: {{output}}",
    ),
]


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

BASE = (
    os.environ.get("LANGFUSE_BASE_URL")
    or os.environ.get("LANGFUSE_HOST")
    or "https://cloud.langfuse.com"
).rstrip("/")
PUB = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
SEC = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()


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


def project_eval_names() -> set[str]:
    _, body = lf_request("GET", "/api/public/unstable/evaluators?limit=100")
    return {
        str(e.get("name"))
        for e in (body.get("data") or [])
        if e.get("scope") == "project" and e.get("name")
    }


def rule_names() -> set[str]:
    _, body = lf_request("GET", "/api/public/unstable/evaluation-rules?limit=100")
    return {str(r.get("name")) for r in (body.get("data") or []) if r.get("name")}


def ensure_score_configs(report: dict[str, Any]) -> None:
    created: list[str] = []
    skipped: list[str] = []
    _, existing = lf_request("GET", "/api/public/score-configs?limit=100")
    have = {c.get("name") for c in (existing.get("data") or [])}
    for name in SCORE_CONFIGS:
        if name in have:
            skipped.append(name)
            continue
        code, body = lf_request(
            "POST",
            "/api/public/score-configs",
            {
                "name": name,
                "dataType": "NUMERIC",
                "minValue": 0,
                "maxValue": 1,
                "description": f"bench-cockpit {name}",
            },
        )
        if code < 300:
            created.append(name)
        else:
            report.setdefault("errors", []).append(f"score-config {name}: {code} {body}")
    report["score_configs"] = {"created": created, "skipped": skipped}


def ensure_dataset(report: dict[str, Any]) -> str | None:
    _, body = lf_request("GET", "/api/public/v2/datasets?limit=50")
    for d in body.get("data") or []:
        if d.get("name") == DATASET_NAME:
            report["dataset"] = {"id": d.get("id"), "name": DATASET_NAME, "created": False}
            return str(d.get("id"))
    code, created = lf_request(
        "POST",
        "/api/public/v2/datasets",
        {
            "name": DATASET_NAME,
            "description": "V5 ablation cells for Langfuse experiments",
        },
    )
    if code < 300:
        report["dataset"] = {
            "id": created.get("id"),
            "name": DATASET_NAME,
            "created": True,
        }
        return str(created.get("id"))
    report.setdefault("errors", []).append(f"dataset: {code} {created}")
    return None


def ensure_llm_connection(report: dict[str, Any]) -> bool:
    code, body = lf_request("GET", "/api/public/llm-connections?limit=20")
    conns = body.get("data") or []
    report["llm_connections"] = [
        {
            "provider": c.get("provider"),
            "adapter": c.get("adapter"),
            "baseURL": c.get("baseURL"),
            "customModels": c.get("customModels"),
        }
        for c in conns
    ]
    ok = any(
        c.get("provider") == PROVIDER
        and MODEL in (c.get("customModels") or [])
        for c in conns
    )
    if not ok:
        report.setdefault("errors", []).append(
            f"Missing LLM connection provider={PROVIDER} model={MODEL}. "
            "Add it in Langfuse → Settings → LLM Connections "
            "(adapter anthropic, base https://api.minimax.io/anthropic)."
        )
    return ok


def create_evaluator_with_retry(name: str, prompt: str, retries: int = 6) -> tuple[bool, Any]:
    have = project_eval_names()
    if name in have:
        return True, {"skipped": True, "name": name}
    last: Any = None
    for i in range(retries):
        code, body = lf_request(
            "POST",
            "/api/public/unstable/evaluators",
            {
                "name": name,
                "prompt": prompt,
                "outputDefinition": OUTDEF,
                "modelConfig": {"provider": PROVIDER, "model": MODEL},
            },
        )
        last = body
        if code in (200, 201, 409):
            return True, body
        # flaky Minimax structured-output preflight
        time.sleep(1.5 + i * 0.5)
    return False, last


def ensure_evaluators(report: dict[str, Any]) -> list[str]:
    ok_names: list[str] = []
    for name, prompt in RUBRICS:
        ok, body = create_evaluator_with_retry(name, prompt)
        if ok:
            ok_names.append(name)
            report.setdefault("evaluators", {}).setdefault("ok", []).append(name)
        else:
            report.setdefault("errors", []).append(f"evaluator {name}: {body}")
    return ok_names


def create_rule(
    name: str,
    evaluator: str,
    target: str,
    *,
    dataset_id: str | None = None,
    enabled: bool = True,
) -> tuple[int, Any]:
    mapping = [
        {"variable": "input", "source": "input"},
        {"variable": "output", "source": "output"},
    ]
    payload: dict[str, Any] = {
        "name": name,
        "evaluator": {"name": evaluator, "scope": "project"},
        "target": target,
        "enabled": enabled,
        "sampling": 1.0,
        "mapping": mapping,
    }
    if target == "observation":
        payload["filter"] = [
            {
                "type": "stringOptions",
                "column": "type",
                "operator": "any of",
                "value": ["GENERATION"],
            }
        ]
    elif target == "experiment" and dataset_id:
        payload["filter"] = [
            {
                "type": "stringOptions",
                "column": "datasetId",
                "operator": "any of",
                "value": [dataset_id],
            }
        ]
    return lf_request("POST", "/api/public/unstable/evaluation-rules", payload)


def ensure_rules(report: dict[str, Any], evaluators: list[str], dataset_id: str | None) -> None:
    existing = rule_names()
    created: list[str] = []
    for ev in evaluators:
        for target, prefix in (("observation", "obs"), ("experiment", "exp")):
            rule = f"{prefix}-{ev}"
            if rule in existing:
                report.setdefault("rules", {}).setdefault("skipped", []).append(rule)
                continue
            if target == "experiment" and not dataset_id:
                continue
            code, body = create_rule(rule, ev, target, dataset_id=dataset_id, enabled=True)
            if code < 300:
                created.append(rule)
                continue
            # preflight flake: create disabled then enable
            code2, body2 = create_rule(
                rule, ev, target, dataset_id=dataset_id, enabled=False
            )
            if code2 >= 300:
                report.setdefault("errors", []).append(f"rule {rule}: {code} {body}")
                continue
            rid = body2.get("id")
            if rid:
                code3, body3 = lf_request(
                    "PATCH",
                    f"/api/public/unstable/evaluation-rules/{rid}",
                    {"enabled": True},
                )
                if code3 >= 300:
                    report.setdefault("errors", []).append(
                        f"enable {rule}: {code3} {body3}"
                    )
                else:
                    created.append(rule)
            else:
                created.append(rule)
    report.setdefault("rules", {})["created"] = created


def status() -> dict[str, Any]:
    out: dict[str, Any] = {"base": BASE, "provider": PROVIDER, "model": MODEL}
    ensure_llm_connection(out)
    _, evals = lf_request("GET", "/api/public/unstable/evaluators?limit=100")
    out["project_evaluators"] = [
        {
            "name": e.get("name"),
            "modelConfig": e.get("modelConfig"),
            "evaluationRuleCount": e.get("evaluationRuleCount"),
        }
        for e in (evals.get("data") or [])
        if e.get("scope") == "project"
    ]
    _, rules = lf_request("GET", "/api/public/unstable/evaluation-rules?limit=100")
    out["rules"] = [
        {
            "name": r.get("name"),
            "target": r.get("target"),
            "enabled": r.get("enabled"),
            "status": r.get("status"),
        }
        for r in (rules.get("data") or [])
    ]
    _, scores = lf_request("GET", "/api/public/score-configs?limit=100")
    out["score_configs"] = [c.get("name") for c in (scores.get("data") or [])]
    return out


def sync() -> dict[str, Any]:
    report: dict[str, Any] = {
        "base": BASE,
        "provider": PROVIDER,
        "model": MODEL,
    }
    if not ensure_llm_connection(report):
        return report
    ensure_score_configs(report)
    dataset_id = ensure_dataset(report)
    evaluators = ensure_evaluators(report)
    ensure_rules(report, evaluators, dataset_id)
    report["status"] = status()
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="List current judges/rules only")
    args = ap.parse_args()
    if args.status:
        print(json.dumps(status(), indent=2))
        return
    print(json.dumps(sync(), indent=2))


if __name__ == "__main__":
    main()
