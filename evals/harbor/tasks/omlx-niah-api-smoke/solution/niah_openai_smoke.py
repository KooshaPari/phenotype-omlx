#!/usr/bin/env python3
"""OpenAI-compatible NIAH smoke used by Harbor task + host dry-run.

Calls ``OPENAI_BASE_URL`` /v1/chat/completions with a fixed needle prompt.
Model defaults from ``omlx_research.smoke_models`` (Qwen3.5 SSOT).

Env:
  OPENAI_BASE_URL   required OpenAI-compatible base (…/v1). Self-host OK; pick a free port.
                    Example: http://127.0.0.1:8766/v1 — no silent skip
  OPENAI_API_KEY    optional (default ``omlx``)
  OMLX_READY_MODEL / smoke_models role=niah for model id
  OMLX_NIAH_OUT     optional path to write JSON result (default stdout)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

NEEDLE = "42-alpha"
SHORT_PROMPT = (
    "Read the passage and reply with ONLY the secret code (no thinking, no quotes).\n\n"
    f"Passage: Notes about weather and tea. The secret code is {NEEDLE}. "
    "More filler about cats.\n\n"
    "Secret code:"
)


def build_prompt(context_tokens: int) -> str:
    """Build the short smoke or an exact Qwen3.5-token 8192-token prompt."""
    if context_tokens <= 0:
        return SHORT_PROMPT
    intro = "Read the following passage and reply with ONLY the secret code.\n\n"
    needle = f"Important context: {NEEDLE}. This fact is critical."
    fixed_tokens = 26
    chat_template_overhead = 25
    filler_tokens = context_tokens - fixed_tokens - chat_template_overhead
    if filler_tokens <= 0:
        raise SystemExit(f"error: NIAH_CONTEXT_TOKENS too small: {context_tokens}")
    before = (filler_tokens * 3) // 4
    return intro + (" the" * before) + needle + (" the" * (filler_tokens - before))


def _extract_reply(payload: dict) -> str:
    """Pull assistant text from OpenAI-compat payloads (incl. reasoning-only)."""
    try:
        msg = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    if not isinstance(msg, dict):
        return str(msg or "")
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content
    # Qwen3.5 / thinking models may put text in reasoning / reasoning_content
    for key in ("reasoning_content", "reasoning", "text"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _model_id() -> str:
    override = os.environ.get("OMLX_READY_MODEL", "").strip()
    if override:
        return override
    # Host / Harbor may not have omlx on PYTHONPATH — allow OPENAI_MODEL
    env_model = os.environ.get("OPENAI_MODEL", "").strip()
    if env_model:
        return env_model
    try:
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "python"))
        from omlx_research.smoke_models import default_model_for

        return default_model_for("niah")
    except Exception as e:
        raise SystemExit(
            f"error: cannot resolve Qwen3.5 model id ({e}); "
            "set OMLX_READY_MODEL or OPENAI_MODEL"
        ) from e


def run_niah() -> dict:
    base = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
    if not base:
        raise SystemExit(
            "error: OPENAI_BASE_URL required (OpenAI-compatible omlx / vLLM). "
            "Do not fall back — mature evals fail loud."
        )
    if base.endswith("/chat/completions"):
        url = base
    elif base.endswith("/v1"):
        url = base + "/chat/completions"
    else:
        url = base + "/v1/chat/completions"

    model = _model_id()
    lower = model.lower()
    if "qwen2.5" in lower:
        raise SystemExit(f"error: Qwen2.5 quarantined (got {model!r})")
    if "qwen3.5" not in lower:
        raise SystemExit(f"error: NIAH smoke requires Qwen3.5 model id (got {model!r})")

    key = os.environ.get("OPENAI_API_KEY", "omlx")
    requested_tokens = int(os.environ.get("NIAH_CONTEXT_TOKENS", "0"))
    prompt = build_prompt(requested_tokens)
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Reply with only the secret code. No analysis.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        # mlx-lm uses a request with a seed as the deterministic sequential
        # path. This avoids the 0.31.2 BatchGenerator worker-stream crash and
        # keeps this one-request NIAH oracle reproducible.
        "seed": 0,
        "max_tokens": 128,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    t0 = datetime.now(timezone.utc)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise SystemExit(f"error: NIAH request failed: {e}") from e

    text = _extract_reply(payload)
    if not text:
        text = json.dumps(payload)[:500]

    hit = NEEDLE in (text or "")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    prompt_tokens = usage.get("prompt_tokens")
    result = {
        "kind": "omlx_niah_api_smoke",
        "ts": t0.isoformat(),
        "model": model,
        "openai_base_url": base,
        "needle": NEEDLE,
        "requested_context_tokens": requested_tokens or None,
        "chat_template_overhead_tokens": 25 if requested_tokens else None,
        "prompt_tokens": prompt_tokens,
        "context_tokens_exact": requested_tokens > 0 and prompt_tokens == requested_tokens,
        "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
        "reply": text,
        "exact_match": hit,
        "evidence_class": "live_api" if hit else "live_api_miss",
    }
    return result


def main() -> int:
    result = run_niah()
    out = os.environ.get("OMLX_NIAH_OUT", "").strip()
    blob = json.dumps(result, indent=2) + "\n"
    if out:
        Path(out).write_text(blob)
    else:
        sys.stdout.write(blob)
    # Also write Harbor-friendly answer file when /app exists
    app_ans = Path("/app/niah_answer.txt")
    if Path("/app").is_dir():
        app_ans.write_text((result.get("reply") or "").strip() + "\n")
        Path("/app/niah_result.json").write_text(blob)
    return 0 if result["exact_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
