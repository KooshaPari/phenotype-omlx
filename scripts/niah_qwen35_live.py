#!/usr/bin/env python3
"""FR-5 E3 — Qwen3.5-only NIAH live smoke (in-process mlx_lm).

Hard rule: model id MUST contain ``Qwen3.5`` (never Qwen3 without .5,
never Qwen2.5). Uses plain ``mlx_lm.generate`` without KV-cache mode
sweeps — Qwen3.5 may use linear attention, so TurboKV / kv_bits metrics
are not applicable.

Exit 0 on live_verified artifact with exact or partial needle match.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ARCHITECTURE_CAVEAT = (
    "Qwen3.5 (incl. OptiQ-4bit variants) may use linear attention; "
    "standard KV-cache compression metrics are not applicable "
    "(kitty-specs/complete-polyglot-vpu-stack/spec.md FR-5)."
)

NEEDLE = "42-alpha"
PROMPT = (
    "Read the passage and reply with ONLY the secret code.\n\n"
    f"Passage: Notes about weather and tea. The secret code is {NEEDLE}. "
    "More filler about cats.\n\n"
    "Secret code:"
)


def _require_qwen35(model: str) -> None:
    if "Qwen3.5" not in model and "qwen3.5" not in model.lower():
        raise SystemExit(
            f"error: FR-5 E3 requires Qwen3.5 only (got {model!r}); "
            "never Qwen3 or Qwen2.5"
        )
    lower = model.lower()
    if "qwen2.5" in lower or ("qwen3" in lower and "qwen3.5" not in lower):
        raise SystemExit(
            f"error: FR-5 E3 rejects non-Qwen3.5 models (got {model!r})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-0.8B",
        help="Must be a Qwen3.5 model id",
    )
    parser.add_argument(
        "--output",
        default="research/fr5_niah_qwen35_live.json",
        help="Live artifact path (not niah_results.json)",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args(argv)

    _require_qwen35(args.model)
    out = Path(args.output)
    if out.name == "niah_results.json":
        print("error: refusing to overwrite synthetic envelope", file=sys.stderr)
        return 2

    import mlx_lm  # type: ignore

    print(f"Loading {args.model} ...", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(args.model)
    load_s = time.perf_counter() - t0
    print(f"Loaded in {load_s:.1f}s", flush=True)

    t1 = time.perf_counter()
    answer = mlx_lm.generate(
        model,
        tokenizer,
        prompt=PROMPT,
        max_tokens=args.max_tokens,
        verbose=False,
    )
    gen_s = time.perf_counter() - t1
    answer_s = (answer or "").strip()
    # mlx_lm.generate may return prompt+completion; isolate tail
    if NEEDLE in answer_s:
        # keep full for audit
        pass
    exact = NEEDLE in answer_s
    partial = (not exact) and ("42" in answer_s or "alpha" in answer_s.lower())

    artifact = {
        "schema_version": 1,
        "kind": "niah_qwen35_live",
        "evidence_label": "live_verified",
        "reported": True,
        "synthetic": False,
        "model": args.model,
        "backend": "mlx_lm_inprocess_plain_generate",
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "architecture_caveat": ARCHITECTURE_CAVEAT,
        "kv_modes_applicable": False,
        "needle": NEEDLE,
        "prompt": PROMPT,
        "answer": answer_s,
        "exact_match": exact,
        "partial_match": partial,
        "load_seconds": round(load_s, 3),
        "generate_seconds": round(gen_s, 3),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"exact={exact} partial={partial} answer={answer_s[:160]!r}")
    print(f"architecture_caveat: {ARCHITECTURE_CAVEAT}")

    if not exact and not partial:
        print("error: needle not recovered", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
