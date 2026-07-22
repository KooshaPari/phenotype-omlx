#!/usr/bin/env python3
"""FR-5 E3 — read-only NIAH needle smoke against a running mlx_lm.server.

DEPRECATED for operator runs — prefer:

    bash scripts/evals/run_via_harbor.sh --niah

(with ``OPENAI_BASE_URL``). This script may still probe an already-loaded
server for FR-5 doctor paths; do not steal ``:8765`` without ownership.

Reuses an already-loaded model (default http://127.0.0.1:8765) so we do
not spawn a second weight load. Writes a labeled live artifact; never
overwrites the committed synthetic ``niah_results.json`` envelope.

Exit codes:
  0 — live_verified artifact written (needle recovered or partial)
  2 — server unreachable / bad response / missing needle
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ARCHITECTURE_CAVEAT = (
    "Qwen3.5 (incl. OptiQ-4bit variants) may use linear attention; "
    "standard KV-cache compression metrics are not applicable "
    "(kitty-specs/complete-polyglot-vpu-stack/spec.md FR-5)."
)

DEFAULT_NEEDLE = "42-alpha"
DEFAULT_PROMPT = (
    "You are a careful assistant. In the haystack below, find the secret "
    f"code and reply with ONLY the code.\n\n"
    f"Haystack: The weather is mild. Notes: laundry Tuesday. "
    f"The secret code is {DEFAULT_NEEDLE}. More filler about cats and tea.\n\n"
    "Secret code:"
)


def chat_completion(base_url: str, model: str, prompt: str, timeout: float) -> dict:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8765",
        help="mlx_lm.server base URL (default: http://127.0.0.1:8765)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-0.8B",
        help="Model id as advertised by the server",
    )
    parser.add_argument(
        "--output",
        default="research/niah_qwen35_server_smoke.json",
        help="Live artifact path (must not be niah_results.json)",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    out = Path(args.output)
    if out.name == "niah_results.json":
        print(
            "error: refusing to overwrite committed synthetic envelope "
            "niah_results.json",
            file=sys.stderr,
        )
        return 2

    try:
        payload = chat_completion(
            args.server_url, args.model, DEFAULT_PROMPT, args.timeout
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"error: server call failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    try:
        answer = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"error: unexpected response shape: {e}", file=sys.stderr)
        return 2

    answer_s = (answer or "").strip()
    exact = DEFAULT_NEEDLE in answer_s
    partial = "42" in answer_s or "alpha" in answer_s.lower()

    artifact = {
        "schema_version": 1,
        "kind": "niah_server_smoke",
        "evidence_label": "live_verified",
        "reported": True,
        "synthetic": False,
        "model": args.model,
        "backend": "mlx_lm_server",
        "server_url": args.server_url,
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "architecture_caveat": ARCHITECTURE_CAVEAT,
        "needle": DEFAULT_NEEDLE,
        "prompt": DEFAULT_PROMPT,
        "answer": answer_s,
        "exact_match": exact,
        "partial_match": partial and not exact,
        "raw_response": payload,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} evidence_label=live_verified exact={exact} partial={partial}")
    print(f"answer: {answer_s[:120]!r}")
    print(f"architecture_caveat: {ARCHITECTURE_CAVEAT}")

    if not exact and not partial:
        print("error: needle not recovered (exact or partial)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
