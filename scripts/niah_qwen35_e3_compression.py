#!/usr/bin/env python3
"""Emit Qwen3.5-only FR-5 E3 metrics from paired in-process cache prefills.

This is intentionally separate from Harbor's OpenAI smoke: only an in-process
run owns the MLX cache objects required to prove state compression.  A failed
or inapplicable contract writes a nonqualifying envelope and exits nonzero.
It never converts absent state into zero-byte compression evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from omlx_research.benchmarks.qwen35_e3_runner import (  # noqa: E402
    PairedPrefillDeps,
    run_paired_prefill,
)
from omlx_research.benchmarks.qwen35_turbokv_overlay import (  # noqa: E402
    make_qwen35_e3_turbokv_cache,
)
from omlx_research.benchmarks.qwen35_state_compression import (  # noqa: E402
    CompressionContractError,
)


NEEDLE = "fr5-e3-qwen35-cache-state"


def _require_qwen35(model_id: str) -> None:
    lower = model_id.lower()
    if "qwen3.5" not in lower or "qwen2.5" in lower:
        raise ValueError(f"FR-5 E3 requires Qwen3.5 only, got {model_id!r}")


def _envelope(model_id: str, context_tokens: int, *, metrics: dict[str, Any] | None,
              error: str | None) -> dict[str, Any]:
    qualified = metrics is not None and error is None
    return {
        "schema_version": 1,
        "kind": "niah_qwen35_e3_runtime_state",
        "evidence_label": "live_verified" if qualified else "not_applicable",
        "reported": qualified,
        "synthetic": False,
        "model": model_id,
        "backend": "mlx_lm_inprocess_paired_prefill",
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evidence_scope": "qwen35_full_attention_runtime_state_only",
        "context_tokens": context_tokens,
        "e3_compression_qualifying": qualified,
        "metrics": metrics,
        "not_applicable_reason": error,
        "non_claims": [
            "Does not prove linear-recurrent state compression.",
            "Does not claim Harbor/OpenAI cache visibility.",
        ],
    }


def _exact_prompt_ids(tokenizer: Any, target_len: int) -> list[int]:
    """Construct an exact-token NIAH prefill without importing MLX."""

    intro = list(tokenizer.encode("Read and retain the critical fact.\n"))
    needle = list(tokenizer.encode(f" Critical fact: {NEEDLE}.\n"))
    filler = list(tokenizer.encode(" the"))
    if not filler or target_len <= len(intro) + len(needle):
        raise ValueError("context_tokens is too short for exact E3 NIAH framing")
    fill_len = target_len - len(intro) - len(needle)
    repeated = (filler * ((fill_len // len(filler)) + 1))[:fill_len]
    before = (fill_len * 3) // 4
    return intro + repeated[:before] + needle + repeated[before:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-OptiQ-4bit")
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--key-bits", type=int, default=4)
    parser.add_argument("--output", default="research/fr5_niah_qwen35_e3_runtime_state.json")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.name == "niah_results.json":
        raise SystemExit("refusing to overwrite synthetic NIAH envelope")

    try:
        _require_qwen35(args.model)
        import mlx_lm  # type: ignore
        from mlx_lm.models.cache import make_prompt_cache  # type: ignore
        from mlx.nn.layers.turbo_kv_cache import (  # type: ignore
            compact_turbo_cache,
        )

        loaded: tuple[Any, Any] | None = None

        def load_once(model_id: str) -> tuple[Any, Any]:
            nonlocal loaded
            if loaded is None:
                loaded = mlx_lm.load(model_id)
            return loaded

        # Obtain tokenizer through the same memoized load that the runner uses.
        _, tokenizer = load_once(args.model)
        prompt_ids = _exact_prompt_ids(tokenizer, args.context_tokens)
        metrics = run_paired_prefill(
            model_id=args.model,
            prompt_ids=prompt_ids,
            deps=PairedPrefillDeps(
                load_model=load_once,
                make_fp16_cache=make_prompt_cache,
                make_turbo_cache=lambda model: make_qwen35_e3_turbokv_cache(
                    model, bits=args.bits, key_bits=args.key_bits
                ),
                generate=mlx_lm.generate,
                compact_turbo_cache=compact_turbo_cache,
            ),
        )
        document = _envelope(args.model, args.context_tokens, metrics=metrics, error=None)
    except (CompressionContractError, ImportError, ValueError, RuntimeError) as exc:
        document = _envelope(
            args.model,
            args.context_tokens,
            metrics=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} qualifying={document['e3_compression_qualifying']}")
    return 0 if document["e3_compression_qualifying"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
