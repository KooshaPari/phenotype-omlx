"""Shared runtime helpers for the real-model harness."""

from __future__ import annotations

import os
import subprocess


def rss_mb() -> float:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).decode().strip()
    return int(out) / 1024


_peak = rss_mb()


def peak_rss_mb() -> float:
    global _peak
    _peak = max(_peak, rss_mb())
    return _peak


def num_layers(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "layers"):
        return len(model.layers)
    raise RuntimeError("Cannot discover num_layers")


def eval_cache(cache_list, mx) -> None:
    arrays = []
    for cache in cache_list:
        state = cache.state
        arrays.extend(state if isinstance(state, (list, tuple)) else [state])
    mx.eval(*arrays)


def kv_bytes(cache_list) -> int:
    total = 0
    for cache in cache_list:
        state = cache.state
        for array in state if isinstance(state, (list, tuple)) else (state,):
            total += array.nbytes if hasattr(array, "nbytes") else 0
    return total


def measure_kv_for_mode(model, tok, prompt, cache_factory, mx):
    """Prefill one cache and return its materialized KV byte count and cache."""

    cache = cache_factory(num_layers(model))
    logits = model(mx.array(tok.encode(prompt))[None], cache=cache)[:, -1, :]
    mx.eval(logits)
    eval_cache(cache, mx)
    return kv_bytes(cache), cache
