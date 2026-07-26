"""Mixed-cache contract tests for the Qwen3.5-only E3 TurboKV overlay."""

from __future__ import annotations

import pytest

from omlx_research.benchmarks.qwen35_turbokv_overlay import (
    Qwen35TurboKVBindings,
    make_qwen35_e3_turbokv_cache,
)


class KVCache:
    pass


class RecurrentState:
    pass


class TurboKVCacheLite:
    def __init__(self, cache: KVCache, *, bits: int, key_bits: int, seed: int) -> None:
        self.cache = cache
        self.bits = bits
        self.key_bits = key_bits
        self.seed = seed


def test_wraps_only_actual_kv_cache_entries_and_preserves_recurrent_identity() -> None:
    recurrent_left = RecurrentState()
    kv = KVCache()
    recurrent_right = RecurrentState()
    bindings = Qwen35TurboKVBindings(
        make_prompt_cache=lambda _model: [recurrent_left, kv, recurrent_right],
        kv_cache_type=KVCache,
        turbo_cache_type=TurboKVCacheLite,
    )

    cache = make_qwen35_e3_turbokv_cache(object(), bits=3, key_bits=2, seed=11, bindings=bindings)

    assert cache[0] is recurrent_left
    assert cache[2] is recurrent_right
    assert isinstance(cache[1], TurboKVCacheLite)
    assert cache[1].cache is kv
    assert (cache[1].bits, cache[1].key_bits, cache[1].seed) == (3, 2, 11)


def test_fails_closed_when_model_cache_has_no_actual_kv_entries() -> None:
    bindings = Qwen35TurboKVBindings(
        make_prompt_cache=lambda _model: [RecurrentState(), RecurrentState()],
        kv_cache_type=KVCache,
        turbo_cache_type=TurboKVCacheLite,
    )

    with pytest.raises(RuntimeError, match="no mlx_lm KVCache entries"):
        make_qwen35_e3_turbokv_cache(object(), bindings=bindings)


def test_fails_closed_when_runtime_binding_cannot_be_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "omlx_research.benchmarks.qwen35_turbokv_overlay.resolve_qwen35_turbokv_bindings",
        lambda: (_ for _ in ()).throw(RuntimeError("binding unavailable")),
    )

    with pytest.raises(RuntimeError, match="binding unavailable"):
        make_qwen35_e3_turbokv_cache(object())
