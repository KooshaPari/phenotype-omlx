#!/usr/bin/env python3
"""Push StructuredPrompts to LangSmith Hub and register hosted LLM evaluators.

UI prerequisite (already done by operator):
  Settings → Model configurations → OpenAI Compatible Endpoint
    base_url=https://api.minimax.io/v1
    model=MiniMax-M3
    API key secret name=MINIMAX_API_KEY
  Feature Access → Evaluators → enable that configuration

This script does the API half:
  1. Push StructuredPrompt repos (correctness / hallucination / code_checker)
  2. POST /v1/platform/evaluators type=llm with prompt_repo_handle
  3. Optionally attach run rules on the tracing project via /api/v1/runs/rules

Usage:
  set -a; source .env; set +a
  .venv-evals/bin/python scripts/evals/setup_hosted_judges.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
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

API = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").rstrip("/")
KEY = os.environ.get("LANGSMITH_API_KEY", "").strip()
PROJECT = os.environ.get("LANGSMITH_PROJECT", "bench-cockpit")
TENANT = os.environ.get("LANGSMITH_TENANT_ID", "").strip()

# Hub handles (workspace-local)
PROMPTS: list[tuple[str, str]] = [
    (
        "bench-correctness",
        "Score 1 only if the reply correctly solves the task; 0 if it echoes the prompt, "
        "repeats blocks, or fails the task. Partial credit 0.3-0.7 for partial solutions.",
    ),
    (
        "bench-hallucination",
        "Score 1 if the reply does not invent APIs/paths/facts; 0 if it fabricates. "
        "Echoing instructions is hallucination of competence — score low.",
    ),
    (
        "bench-code-checker",
        "If the task needs code/bash/diff, score whether the code would run/solve it. "
        "Repeated identical snippets score 0. Score non-code tasks as 0.5.",
    ),
]

SCHEMA = {
    "title": "JudgeScore",
    "description": "LLM-as-judge score for bench-cockpit",
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "description": "0.0–1.0 quality score for the rubric",
        },
        "reason": {
            "type": "string",
            "description": "Short justification",
        },
    },
    "required": ["score"],
}


def die(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def ls_request(method: str, path: str, body: Any | None = None) -> tuple[int, Any]:
    if not KEY:
        die("LANGSMITH_API_KEY required")
    data = None if body is None else json.dumps(body).encode()
    headers = {
        "x-api-key": KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if TENANT:
        headers["x-tenant-id"] = TENANT
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers=headers,
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


def ensure_tenant() -> str:
    global TENANT
    if TENANT:
        return TENANT
    code, body = ls_request("GET", "/workspaces")
    if code >= 300:
        die(f"workspaces {code}: {body}")
    ws = body if isinstance(body, list) else body.get("workspaces") or []
    if not ws:
        die("no LangSmith workspaces")
    TENANT = str(ws[0]["id"])
    os.environ["LANGSMITH_TENANT_ID"] = TENANT
    return TENANT


def push_prompts() -> dict[str, str]:
    """Push StructuredPrompts via langsmith SDK (requires .venv-evals)."""
    try:
        from langsmith import Client
        from langchain_core.prompts.structured import StructuredPrompt
    except ImportError as e:
        die(
            f"langsmith/langchain-core missing ({e}). "
            "Create apps/bench-cockpit/.venv-evals with python3.12 and install deps."
        )

    client = Client(api_key=KEY, api_url=API)
    urls: dict[str, str] = {}
    for handle, rubric in PROMPTS:
        prompt = StructuredPrompt.from_messages_and_schema(
            [
                (
                    "system",
                    "You are a strict evaluation judge for LLM agent outputs. "
                    "Return structured score/reason only.",
                ),
                (
                    "human",
                    "Rubric: "
                    + rubric
                    + "\n\nInput:\n{input}\n\nOutput:\n{output}\n",
                ),
            ],
            schema=SCHEMA,
        )
        url = client.push_prompt(handle, object=prompt)
        urls[handle] = str(url)
        print(f"pushed {handle}: {url}")
    return urls


def list_evaluators() -> list[dict[str, Any]]:
    code, body = ls_request("GET", "/v1/platform/evaluators")
    if code >= 300:
        die(f"list evaluators {code}: {body}")
    return body.get("evaluators") or []


def upsert_llm_evaluator(handle: str) -> dict[str, Any]:
    existing = {e.get("name"): e for e in list_evaluators()}
    name = handle  # e.g. bench-correctness
    if name in existing and existing[name].get("type") == "llm":
        print(f"evaluator exists: {name} ({existing[name].get('id')})")
        return existing[name]
    payload = {
        "name": name,
        "type": "llm",
        "llm_evaluator": {
            "prompt_repo_handle": handle,
            "commit_hash_or_tag": "latest",
            "variable_mapping": {
                "input": "input",
                "output": "output",
            },
        },
    }
    code, body = ls_request("POST", "/v1/platform/evaluators", payload)
    if code not in (200, 201):
        die(f"create evaluator {name} {code}: {body}")
    ev = body.get("evaluator") or body
    print(f"created evaluator {name}: {ev.get('id')}")
    return ev


def find_project_id(name: str) -> str | None:
    code, body = ls_request("GET", "/sessions?limit=100")
    if code >= 300:
        die(f"sessions {code}: {body}")
    sessions = body if isinstance(body, list) else body.get("sessions") or []
    for s in sessions:
        if s.get("name") == name and not s.get("reference_dataset_id"):
            return s.get("id")
    for s in sessions:
        if s.get("name") == name:
            return s.get("id")
    return None


def attach_run_rule(session_id: str, handle: str, display: str) -> dict[str, Any] | None:
    """Best-effort: attach hub LLM judge to project via runs/rules.

    Model selection for hosted OpenAI-compat configs is workspace-scoped;
    if the rule API rejects the model block, the evaluator still exists in the
    catalog for UI attachment / sampling.
    """
    payload = {
        "display_name": display,
        "session_id": session_id,
        "sampling_rate": 1.0,
        "is_enabled": True,
        "evaluators": [
            {
                "structured": {
                    "hub_ref": f"{handle}:latest",
                    # ChatOpenAI with workspace secret; base_url for Minimax OpenAI-compat
                    "model": {
                        "lc": 1,
                        "type": "constructor",
                        "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
                        "name": "ChatOpenAI",
                        "kwargs": {
                            "model_name": os.environ.get("MINIMAX_JUDGE_MODEL", "MiniMax-M3"),
                            "temperature": 0,
                            "openai_api_base": "https://api.minimax.io/v1",
                            "openai_api_key": {
                                "lc": 1,
                                "type": "secret",
                                "id": ["MINIMAX_API_KEY"],
                            },
                        },
                    },
                }
            }
        ],
    }
    code, body = ls_request("POST", "/api/v1/runs/rules", payload)
    if code not in (200, 201):
        # Fallback path without /api prefix
        code2, body2 = ls_request("POST", "/runs/rules", payload)
        if code2 not in (200, 201):
            print(f"WARN run rule {handle}: {code} {body} / {code2} {body2}")
            return None
        body = body2
    print(f"run rule attached: {display}")
    return body if isinstance(body, dict) else {"raw": body}


def main() -> None:
    ensure_tenant()
    print(f"tenant={TENANT} project={PROJECT}")
    urls = push_prompts()
    created = []
    for handle, _ in PROMPTS:
        created.append(upsert_llm_evaluator(handle))

    pid = find_project_id(PROJECT)
    rules = []
    if pid:
        print(f"project_id={pid}")
        for handle, _ in PROMPTS:
            rules.append(attach_run_rule(pid, handle, f"hosted-{handle}"))
    else:
        print(f"WARN: project {PROJECT!r} not found — evaluators registered, attach rules in UI")

    out = {
        "tenant_id": TENANT,
        "project": PROJECT,
        "project_id": pid,
        "prompt_urls": urls,
        "evaluators": [{"id": e.get("id"), "name": e.get("name"), "type": e.get("type")} for e in created],
        "run_rules_attached": sum(1 for r in rules if r),
        "note": "Ensure Model config MiniMax-M3 is enabled under Feature Access → Evaluators",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
