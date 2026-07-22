"""Local web admin: start a small HTTP server that serves the
phenotype-omlx admin extensions (research panel + REST endpoints).

This is the lightweight "web" tier of the unified launcher — when the user
runs `omlx-research web`, this module binds to a port and serves:

  GET  /                        — research panel HTML
  GET  /static/<file>            — CSS / JS assets
  GET  /api/v1/status           — backend availability JSON
  GET  /api/v1/fleet            — fleet peers JSON
  POST /api/v1/inference        — single-shot inference
  POST /api/v1/spec-decode      — speculative decoding invocation
"""

from __future__ import annotations
import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
from pathlib import Path

PANEL_DIR = Path(__file__).resolve().parents[3] / "gui" / "admin-extensions"


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 — silence default access log
        sys.stderr.write(
            "[omlx-research-web] %s - %s\n" % (self.address_string(), format % args)
        )

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.startswith("/index.html"):
            self._serve_file(
                PANEL_DIR / "templates" / "research_panel.html", "text/html"
            )
        elif self.path.startswith("/static/"):
            rel = self.path[len("/static/") :]
            full = PANEL_DIR / "static" / rel
            ct = (
                "text/css"
                if rel.endswith(".css")
                else "application/javascript"
                if rel.endswith(".js")
                else "text/plain"
            )
            self._serve_file(full, ct)
        elif self.path.startswith("/api/v1/status"):
            self._json(self._status())
        elif self.path.startswith("/api/v1/fleet"):
            self._json(
                {
                    "peers": [],
                    "self": os.uname().nodename if hasattr(os, "uname") else "node",
                }
            )
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/api/v1/inference"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                req = json.loads(body)
            except json.JSONDecodeError:
                self._json({"error": "invalid json"}, 400)
                return
            self._json(
                {
                    "error": "not_implemented",
                    "detail": (
                        "POST /api/v1/inference is a stub. "
                        "Wire to HybridDispatch or implement a backend-specific "
                        "handler to enable real inference."
                    ),
                },
                501,
            )
        elif self.path.startswith("/api/v1/spec-decode"):
            self._json({"mode": "stub", "accepted": [], "acceptance_rate": 0.0})
        else:
            self.send_error(404)

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _status(self) -> dict:
        out: dict = {"backends": {}}
        try:
            from omlx_research.backends import (
                MlxBackend,
                MetalKernelBackend,
                VllmBackend,
                TensorrtBackend,
                SglangBackend,
                LlamaCppBackend,
            )

            for cls in (
                MlxBackend,
                MetalKernelBackend,
                VllmBackend,
                TensorrtBackend,
                SglangBackend,
                LlamaCppBackend,
            ):
                b = cls()
                out["backends"][b.capabilities.name] = {
                    "primary": b.capabilities.primary,
                    "available": b.is_available(),
                }
        except Exception as exc:  # pragma: no cover — diagnostic
            out["error"] = repr(exc)
        return out


class _ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="omlx-research-web")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    srv = _ThreadingServer((args.host, args.port), _Handler)
    print(f"omlx-research web: http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
