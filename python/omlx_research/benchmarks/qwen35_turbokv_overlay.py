"""Qwen3.5 E3-only TurboKV cache wiring.

Qwen3.5 cache lists can mix full-attention ``KVCache`` entries with
linear-recurrent state objects.  The vendor helper is model-generic; this
small boundary makes the E3 contract explicit: only runtime entries that are
actually ``mlx_lm.models.cache.KVCache`` instances are wrapped.  Recurrent
state is retained by identity, so this adapter cannot reinterpret it as KV
state or silently create a compression claim where none applies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Qwen35TurboKVBindings:
    """The minimal runtime boundary, injectable for mixed-cache tests."""

    make_prompt_cache: Callable[[Any], Sequence[Any]]
    kv_cache_type: type[Any]
    turbo_cache_type: Callable[..., Any]


def resolve_qwen35_turbokv_bindings() -> Qwen35TurboKVBindings:
    """Resolve MLX bindings lazily, failing closed when the overlay is absent."""

    try:
        from mlx_lm.models.cache import KVCache, make_prompt_cache  # type: ignore
        from mlx.nn.layers.turbo_kv_cache import TurboKVCacheLite  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3.5 E3 TurboKV overlay is unavailable; set OMLX_TURBOQUANT_LAYER "
            "and ensure MLX/MLX-LM are importable"
        ) from exc
    return Qwen35TurboKVBindings(
        make_prompt_cache=make_prompt_cache,
        kv_cache_type=KVCache,
        turbo_cache_type=TurboKVCacheLite,
    )


def make_qwen35_e3_turbokv_cache(
    model: Any,
    *,
    bits: int = 4,
    key_bits: int = 4,
    seed: int = 42,
    bindings: Qwen35TurboKVBindings | None = None,
) -> list[Any]:
    """Build a mixed Qwen3.5 cache list with TurboKV only on real KV entries.

    This is intentionally scoped to paired E3 prefills.  It does not mutate
    the vendor helper, sitecustomize, or generic backend wiring.  A model with
    no standard KVCache entries is ineligible for this cache-compression
    measurement and raises instead of returning an all-recurrent fake route.
    """

    runtime = bindings or resolve_qwen35_turbokv_bindings()
    cache = list(runtime.make_prompt_cache(model))
    kv_indices = [
        index for index, entry in enumerate(cache) if isinstance(entry, runtime.kv_cache_type)
    ]
    if not kv_indices:
        raise RuntimeError(
            "Qwen3.5 E3 cache is ineligible: make_prompt_cache produced no mlx_lm KVCache entries"
        )
    for index in kv_indices:
        cache[index] = runtime.turbo_cache_type(
            cache[index], bits=bits, key_bits=key_bits, seed=seed
        )
    return cache
