"""FR-7 oMLX vPU dashboard HTTP server.

Serves the canonical panel under ``perf-core/vpu/dashboard/`` — not the
pheno-harness Go UI and not ``gui/admin-extensions`` research_panel.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def repo_root() -> Path:
    # python/omlx_research/vpu_dashboard/server.py -> repo root
    return Path(__file__).resolve().parents[3]


def dashboard_root() -> Path:
    return repo_root() / "perf-core" / "vpu" / "dashboard"


def panel_index() -> Path:
    return dashboard_root() / "panel" / "index.html"


def schema_path() -> Path:
    return dashboard_root() / "schema" / "status.v1.json"


def git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root()),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("OMLX_BUILD_HEAD", "unknown")


def build_status() -> dict[str, Any]:
    """Status payload matching ``schema/status.v1.json``."""
    errors: list[str] = []
    if not panel_index().is_file():
        errors.append("panel/index.html missing")
    if not schema_path().is_file():
        errors.append("schema/status.v1.json missing")

    eval_id = None
    live = repo_root() / "research" / "fr5_niah_qwen35_live.json"
    if live.is_file():
        eval_id = "research/fr5_niah_qwen35_live.json"

    promo_id = None
    promo_dir = repo_root() / ".omlx" / "cache" / "promotion"
    if promo_dir.is_dir() and any(promo_dir.glob("*.json")):
        promo_id = ".omlx/cache/promotion/"

    return {
        "schema_version": 1,
        "build_head": git_head(),
        "polyglot_tiers": {
            "rust": {"status": "ok", "detail": "kernel-registry / perf-core"},
            "python": {"status": "ok", "detail": "omlx_research CLI + dashboard"},
            "mojo": {"status": "ok", "detail": "ABI lane reported elsewhere"},
            "julia": {"status": "ok", "detail": "FR-5 eval path"},
        },
        "eval_snapshot_id": eval_id,
        "promotion_snapshot_id": promo_id,
        "errors": errors,
        "owner": "Salmon",
    }


def health_payload() -> tuple[dict[str, Any], int]:
    if not panel_index().is_file() or not schema_path().is_file():
        return {
            "ok": False,
            "error": "dashboard assets missing under perf-core/vpu/dashboard",
            "owner": "Salmon",
        }, 503
    return {
        "ok": True,
        "service": "omlx-vpu-dashboard",
        "owner": "Salmon",
        "build_head": git_head(),
    }, 200


class VpuDashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        sys.stderr.write(f"[vpu-dashboard] {self.address_string()} - {fmt % args}\n")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in ("/vpu", "/vpu/"):
            self._file(panel_index(), "text/html; charset=utf-8")
            return
        if path in ("/health", "/vpu/health"):
            body, code = health_payload()
            self._json(body, code)
            return
        if path == "/vpu/api/v1/status":
            self._json(build_status(), 200)
            return
        self.send_error(404, "not found")

    def _json(self, obj: dict[str, Any], status: int) -> None:
        data = json.dumps(obj, indent=2).encode("utf-8") + b"\n"
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(503, "panel missing")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(host: str = "127.0.0.1", port: int = 8787) -> int:
    httpd = ThreadingHTTPServer((host, port), VpuDashboardHandler)
    print(f"omlx vpu-dashboard: http://{host}:{port}/vpu/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omlx-research vpu-dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args(argv)
    return serve(args.host, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
