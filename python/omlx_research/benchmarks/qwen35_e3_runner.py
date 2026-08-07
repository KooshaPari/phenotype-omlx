"""In-process paired-prefill orchestration for Qwen3.5 FR-5 E3.

This runner intentionally has no OpenAI/Harbor path: server requests cannot
inspect the server-owned cache objects needed for E3.  Its dependency bundle
makes orchestration unit-testable without importing MLX or loading weights.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .qwen35_state_compression import measure_qwen35_runtime_state_compression


@dataclass(frozen=True)
class PairedPrefillDeps:
    """Runtime factories used by :func:`run_paired_prefill`.

    Production supplies MLX/MLX-LM functions; tests supply fakes.  Cache byte
    metrics are never injected here and are always read by the state extractor.
    """

    load_model: Callable[[str], tuple[Any, Any]]
    make_fp16_cache: Callable[[Any], Sequence[Any]]
    make_turbo_cache: Callable[[Any], Sequence[Any]]
    generate: Callable[..., Any]
    compact_turbo_cache: Callable[[Sequence[Any]], int]


def run_paired_prefill(
    *, model_id: str, prompt_ids: Sequence[int], deps: PairedPrefillDeps
) -> dict[str, Any]:
    """Prefill the identical Qwen3.5 prompt twice and measure cache state.

    The model is loaded exactly once.  Both cache lists are supplied to the
    same generation API with ``max_tokens=1`` so each represents post-prefill
    runtime state.  The compacted route is materialized before extraction.
    Exceptions are deliberately propagated: callers must write a nonqualifying
    envelope rather than invent an E3 number from a partial execution.
    """

    if not prompt_ids:
        raise ValueError("E3 paired prefill requires non-empty token IDs")
    model, tokenizer = deps.load_model(model_id)
    fp16_cache = deps.make_fp16_cache(model)
    turbo_cache = deps.make_turbo_cache(model)
    if len(fp16_cache) != len(turbo_cache):
        raise ValueError("paired E3 cache factories returned different layer counts")

    common = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": list(prompt_ids),
        "max_tokens": 1,
        "verbose": False,
    }
    deps.generate(prompt_cache=fp16_cache, **common)
    deps.generate(prompt_cache=turbo_cache, **common)
    compacted_layers = deps.compact_turbo_cache(turbo_cache)
    metrics = measure_qwen35_runtime_state_compression(
        model_id=model_id,
        model=model,
        fp16_cache=fp16_cache,
        compacted_cache=turbo_cache,
    )
    return {
        **metrics,
        "paired_prefill": True,
        "prefill_prompt_tokens": len(prompt_ids),
        "compaction_calls_reported": compacted_layers,
    }
