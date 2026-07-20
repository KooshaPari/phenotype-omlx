#!/usr/bin/env python3
"""FR-7 D2 — serve smoke for the canonical oMLX vPU dashboard.

Starts the dashboard on an ephemeral port, curls health + status + panel,
writes a labeled artifact, then shuts down.

Exit 0 on success.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.client import HTTPResponse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from omlx_research.vpu_dashboard.server import (  # noqa: E402
    VpuDashboardHandler,
    build_status,
    health_payload,
)
from http.server import ThreadingHTTPServer  # noqa: E402


def _get(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # type: ignore[assignment]
        r: HTTPResponse = resp
        return r.status, r.read()


def main() -> int:
    health_body, health_code = health_payload()
    if health_code != 200:
        print(f"error: assets not ready: {health_body}", file=sys.stderr)
        return 2

    status = build_status()
    required = {
        "schema_version",
        "build_head",
        "polyglot_tiers",
        "eval_snapshot_id",
        "promotion_snapshot_id",
        "errors",
        "owner",
    }
    missing = required - set(status)
    if missing:
        print(f"error: status missing fields: {sorted(missing)}", file=sys.stderr)
        return 2
    if status["owner"] != "Salmon":
        print("error: owner must be Salmon", file=sys.stderr)
        return 2

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), VpuDashboardHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    time.sleep(0.15)

    results: dict = {"ok": True, "base": base, "checks": []}
    try:
        for path, want_substr in (
            ("/vpu/health", b'"ok": true'),
            ("/health", b'"ok": true'),
            ("/vpu/api/v1/status", b'"schema_version": 1'),
            ("/vpu/", b"vPU dashboard"),
        ):
            code, body = _get(base + path)
            ok = code == 200 and want_substr in body
            results["checks"].append({
                "path": path,
                "http_status": code,
                "ok": ok,
            })
            if not ok:
                results["ok"] = False
                print(f"FAIL {path} status={code}", file=sys.stderr)
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"error: {e}", file=sys.stderr)
        results["ok"] = False
    finally:
        httpd.shutdown()
        httpd.server_close()

    out = ROOT / "research" / "fr7_dashboard_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "kind": "fr7_dashboard_smoke",
        "evidence_label": "live_verified",
        "reported": True,
        "synthetic": False,
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "owner": "Salmon",
        "status_example": status,
        "results": results,
    }
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0 if results["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
