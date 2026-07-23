#!/usr/bin/env python3
"""Bootstrap Langfuse Cloud Hobby for bench-cockpit + Phenotype agents.

Primary path until Hobby caps (or a meaningful self-host/fork decision).
Idempotent: creates missing score configs, dataset, prompts, annotation queue,
custom dashboards/widgets, then delegates hosted judges to setup_langfuse_judges.py.

  python3 scripts/evals/setup_langfuse_cloud.py
  python3 scripts/evals/setup_langfuse_cloud.py --status
  python3 scripts/evals/setup_langfuse_cloud.py --skip-judges

Env (gitignored .env or export):
  LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
  LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

WIDGETS: list[dict[str, Any]] = [
    {
        "name": "bench-obs-count",
        "description": "Total observations",
        "view": "observations",
        "dimensions": [],
        "metrics": [{"measure": "count", "agg": "count"}],
        "filters": [],
        "chartType": "NUMBER",
    },
    {
        "name": "bench-obs-over-time",
        "description": "Observations by month",
        "view": "observations",
        "dimensions": [{"field": "startTimeMonth"}],
        "metrics": [{"measure": "count", "agg": "count"}],
        "filters": [],
        "chartType": "LINE_TIME_SERIES",
    },
    {
        "name": "bench-cost-over-time",
        "description": "Total cost by month",
        "view": "observations",
        "dimensions": [{"field": "startTimeMonth"}],
        "metrics": [{"measure": "totalCost", "agg": "sum"}],
        "filters": [],
        "chartType": "LINE_TIME_SERIES",
    },
    {
        "name": "bench-latency-p95-by-model",
        "description": "P95 latency by model",
        "view": "observations",
        "dimensions": [{"field": "providedModelName"}],
        "metrics": [{"measure": "latency", "agg": "p95"}],
        "filters": [],
        "chartType": "HORIZONTAL_BAR",
    },
    {
        "name": "bench-score-avg-by-name",
        "description": "Avg numeric scores by name",
        "view": "scores-numeric",
        "dimensions": [{"field": "name"}],
        "metrics": [{"measure": "value", "agg": "avg"}],
        "filters": [],
        "chartType": "VERTICAL_BAR",
    },
    {
        "name": "bench-score-count-by-name",
        "description": "Score volume by name",
        "view": "scores-numeric",
        "dimensions": [{"field": "name"}],
        "metrics": [{"measure": "count", "agg": "count"}],
        "filters": [],
        "chartType": "VERTICAL_BAR",
    },
]

DASHBOARD_NAME = "bench-cockpit-ops"
PROMPT_NAME = "bench-judge-system"
PROMPT_BODY = (
    "You are a strict bench judge. Score only with evidence from input/output. "
    "Prefer 0 when the model echoes the prompt or fabricates APIs/paths."
)
QUEUE_NAME = "bench-manual-review"
DATASET_NAME = "bench-cockpit-v5-cells"
SCORE_CONFIGS = [
    "gen_ok",
    "partial_credit",
    "correctness",
    "hallucination",
    "code_checker",
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
    or "https://us.cloud.langfuse.com"
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


def names_of(path: str) -> list[str]:
    _, body = lf_request("GET", path)
    return [str(x.get("name")) for x in (body.get("data") or []) if x.get("name")]


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


def ensure_dataset(report: dict[str, Any]) -> None:
    _, body = lf_request("GET", "/api/public/v2/datasets?limit=50")
    for d in body.get("data") or []:
        if d.get("name") == DATASET_NAME:
            report["dataset"] = {"name": DATASET_NAME, "id": d.get("id"), "created": False}
            return
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
            "name": DATASET_NAME,
            "id": created.get("id"),
            "created": True,
        }
    else:
        report.setdefault("errors", []).append(f"dataset: {code} {created}")


def ensure_prompt(report: dict[str, Any]) -> None:
    code, body = lf_request("GET", f"/api/public/v2/prompts/{PROMPT_NAME}")
    if code < 300 and body.get("name"):
        report["prompt"] = {"name": PROMPT_NAME, "created": False}
        return
    code, created = lf_request(
        "POST",
        "/api/public/v2/prompts",
        {
            "name": PROMPT_NAME,
            "type": "text",
            "prompt": PROMPT_BODY,
            "labels": ["production"],
            "config": {"phenotype": "bench-cockpit"},
        },
    )
    if code < 300:
        report["prompt"] = {"name": PROMPT_NAME, "created": True}
    else:
        # already exists under some responses
        if code == 409:
            report["prompt"] = {"name": PROMPT_NAME, "created": False}
        else:
            report.setdefault("errors", []).append(f"prompt: {code} {created}")


def ensure_annotation_queue(report: dict[str, Any]) -> None:
    _, body = lf_request("GET", "/api/public/annotation-queues?limit=20")
    for q in body.get("data") or []:
        if q.get("name") == QUEUE_NAME:
            report["annotation_queue"] = {
                "name": QUEUE_NAME,
                "id": q.get("id"),
                "created": False,
            }
            return
    # Hobby = 1 queue max; attach known score configs when possible
    _, sc = lf_request("GET", "/api/public/score-configs?limit=100")
    score_ids = [
        c["id"]
        for c in (sc.get("data") or [])
        if c.get("name") in SCORE_CONFIGS and c.get("id")
    ]
    code, created = lf_request(
        "POST",
        "/api/public/annotation-queues",
        {
            "name": QUEUE_NAME,
            "description": "Human review for ambiguous bench cells",
            "scoreConfigIds": score_ids[:5],
        },
    )
    if code < 300:
        report["annotation_queue"] = {
            "name": QUEUE_NAME,
            "id": created.get("id"),
            "created": True,
        }
    else:
        report.setdefault("errors", []).append(f"annotation-queue: {code} {created}")


def ensure_widgets(report: dict[str, Any]) -> dict[str, str]:
    _, existing = lf_request("GET", "/api/public/unstable/dashboard-widgets?limit=100")
    have = {w["name"]: w for w in (existing.get("data") or []) if w.get("name")}
    ids: dict[str, str] = {}
    created: list[str] = []
    skipped: list[str] = []
    for spec in WIDGETS:
        name = str(spec["name"])
        if name in have:
            ids[name] = str(have[name]["id"])
            skipped.append(name)
            continue
        code, body = lf_request("POST", "/api/public/unstable/dashboard-widgets", spec)
        if code < 300 and body.get("id"):
            ids[name] = str(body["id"])
            created.append(name)
        else:
            report.setdefault("errors", []).append(f"widget {name}: {code} {body}")
    report["widgets"] = {"created": created, "skipped": skipped}
    return ids


def ensure_dashboard(report: dict[str, Any], widget_ids: dict[str, str]) -> None:
    _, dlist = lf_request("GET", "/api/public/unstable/dashboards?limit=50")
    dash: dict[str, Any] | None = None
    for d in dlist.get("data") or []:
        if d.get("name") == DASHBOARD_NAME:
            dash = d
            break
    if dash is None:
        code, created = lf_request(
            "POST",
            "/api/public/unstable/dashboards",
            {
                "name": DASHBOARD_NAME,
                "description": "Bench cockpit quality, cost, latency (Phenotype)",
            },
        )
        if code >= 300:
            report.setdefault("errors", []).append(f"dashboard: {code} {created}")
            return
        dash = created
        report["dashboard"] = {"name": DASHBOARD_NAME, "id": dash.get("id"), "created": True}
    else:
        report["dashboard"] = {
            "name": DASHBOARD_NAME,
            "id": dash.get("id"),
            "created": False,
        }

    dash_id = str(dash["id"])
    _, placements = lf_request(
        "GET", f"/api/public/unstable/dashboards/{dash_id}/placements"
    )
    placed_widget_ids = set()
    for p in placements.get("data") or placements.get("placements") or []:
        wid = p.get("widgetId") or (p.get("widget") or {}).get("id")
        if wid:
            placed_widget_ids.add(str(wid))

    added: list[str] = []
    for i, spec in enumerate(WIDGETS):
        name = str(spec["name"])
        wid = widget_ids.get(name)
        if not wid or wid in placed_widget_ids:
            continue
        payloads = [
            {
                "type": "widget",
                "widgetId": wid,
                "x": (i % 2) * 6,
                "y": (i // 2) * 4,
                "w": 6,
                "h": 4,
            },
            {"type": "widget", "widgetId": wid},
            {"widgetId": wid},
        ]
        ok = False
        last: Any = None
        for body in payloads:
            code, resp = lf_request(
                "POST",
                f"/api/public/unstable/dashboards/{dash_id}/placements",
                body,
            )
            last = resp
            if code < 300:
                ok = True
                added.append(name)
                break
        if not ok:
            report.setdefault("errors", []).append(
                f"placement {name}: {last}"
            )
    report["placements_added"] = added


def llm_connection_status(report: dict[str, Any]) -> None:
    _, body = lf_request("GET", "/api/public/llm-connections?limit=20")
    conns = [
        {
            "provider": c.get("provider"),
            "adapter": c.get("adapter"),
            "baseURL": c.get("baseURL"),
            "customModels": c.get("customModels"),
        }
        for c in (body.get("data") or [])
    ]
    report["llm_connections"] = conns
    ok = any(
        c.get("provider") == "Minimax" and "Minimax-M3" in (c.get("customModels") or [])
        for c in conns
    )
    if not ok:
        report.setdefault("manual", []).append(
            "Add Minimax LLM connection (adapter=anthropic, "
            "base=https://api.minimax.io/anthropic, model=Minimax-M3) if missing."
        )


def manual_integrations_checklist(report: dict[str, Any]) -> None:
    """UI-only on Hobby — document for operators / agents."""
    report["manual_integrations"] = [
        {
            "id": "slack",
            "where": "Project Settings → Integrations → Slack",
            "why": "Trace/score alerts into Phenotype ops channels",
        },
        {
            "id": "prompt_webhooks",
            "where": "Prompts → Automations / webhooks",
            "why": "Notify when production prompt labels change",
        },
        {
            "id": "monitors",
            "where": "Monitoring → Monitors (Hobby max 2)",
            "why": "Latency/cost/score threshold alerts",
        },
        {
            "id": "posthog",
            "where": "Integrations → PostHog (if product analytics active)",
            "why": "Join LLM quality with product events",
        },
        {
            "id": "otel",
            "where": "SDK / OpenTelemetry exporter → Langfuse OTel endpoint",
            "why": "Org-wide services not using Python/JS Langfuse SDK",
        },
        {
            "id": "mcp",
            "where": "Cursor/Claude MCP → /api/public/mcp",
            "why": "Agent dashboard/prompt CRUD; run print-cursor-mcp-snippet.sh",
        },
    ]


def status_report() -> dict[str, Any]:
    code, health = lf_request("GET", "/api/public/health")
    _, projects = lf_request("GET", "/api/public/projects")
    project_id = None
    project_names: list[str] = []
    for p in projects.get("data") or []:
        if p.get("name"):
            project_names.append(str(p["name"]))
        if project_id is None and p.get("id"):
            project_id = str(p["id"])
    _, dashboards = lf_request("GET", "/api/public/unstable/dashboards?limit=50")
    dash_id = None
    dash_names: list[str] = []
    for d in dashboards.get("data") or []:
        if d.get("name"):
            dash_names.append(str(d["name"]))
        if d.get("name") == DASHBOARD_NAME and d.get("id"):
            dash_id = str(d["id"])
    dashboard_url = None
    if project_id and dash_id:
        dashboard_url = f"{BASE}/project/{project_id}/dashboards/{dash_id}"
    return {
        "base_url": BASE,
        "health": health if code < 300 else {"error": health},
        "projects": project_names,
        "score_configs": names_of("/api/public/score-configs?limit=100"),
        "datasets": names_of("/api/public/v2/datasets?limit=50"),
        "prompts": names_of("/api/public/v2/prompts?limit=50"),
        "dashboards": dash_names,
        "dashboard_url": dashboard_url,
        "widgets": names_of("/api/public/unstable/dashboard-widgets?limit=100"),
        "annotation_queues": names_of("/api/public/annotation-queues?limit=20"),
        "evaluators": names_of("/api/public/unstable/evaluators?limit=100"),
    }


def run_judges() -> dict[str, Any]:
    script = ROOT / "scripts" / "evals" / "setup_langfuse_judges.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    out = (proc.stdout or "")[-2000:]
    err = (proc.stderr or "")[-1000:]
    return {"exit": proc.returncode, "stdout_tail": out, "stderr_tail": err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--skip-judges", action="store_true")
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    args = ap.parse_args()

    if "127.0.0.1" in BASE or "localhost" in BASE:
        print(
            f"warning: LANGFUSE_BASE_URL={BASE} looks like self-host; "
            "Phenotype default is Cloud Hobby until caps/fork.",
            file=sys.stderr,
        )

    if args.status:
        report = status_report()
        llm_connection_status(report)
        manual_integrations_checklist(report)
        print(json.dumps(report, indent=2))
        return

    report: dict[str, Any] = {"base_url": BASE}
    code, health = lf_request("GET", "/api/public/health")
    if code >= 300:
        die(f"Langfuse health failed: {code} {health}")
    report["health"] = health

    ensure_score_configs(report)
    ensure_dataset(report)
    ensure_prompt(report)
    ensure_annotation_queue(report)
    widget_ids = ensure_widgets(report)
    ensure_dashboard(report, widget_ids)
    llm_connection_status(report)
    manual_integrations_checklist(report)

    if not args.skip_judges:
        report["judges"] = run_judges()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(report, indent=2))
        if report.get("errors"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
