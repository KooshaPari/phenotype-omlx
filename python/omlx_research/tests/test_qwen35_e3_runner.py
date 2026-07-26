"""Orchestration tests for the Qwen3.5 paired-prefill E3 runner."""

from __future__ import annotations

import pytest

from omlx_research.benchmarks.qwen35_e3_runner import (
    PairedPrefillDeps,
    run_paired_prefill,
)
from omlx_research.benchmarks.qwen35_state_compression import CompressionContractError


class _Array:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes


class _LinearLayer:
    is_linear = True


class _AttentionLayer:
    is_linear = False


class _Model:
    layers = [_LinearLayer(), _AttentionLayer()]


class KVCache:
    def __init__(self, nbytes: int) -> None:
        self.keys = _Array(nbytes // 2)
        self.values = _Array(nbytes // 2)


class TurboKVCacheLite:
    def __init__(self) -> None:
        self._compacted = True
        self._turbo_packed_keys = _Array(300)
        self._turbo_packed_values = _Array(300)
        self._turbo_key_norms = _Array(50)
        self._turbo_value_norms = _Array(50)

    @property
    def nbytes(self) -> int:
        return 800


def test_paired_prefill_loads_once_and_measures_only_after_both_prefills() -> None:
    calls: list[dict] = []
    model = _Model()
    fp16_cache = [KVCache(40), KVCache(2_000)]
    turbo_cache = [KVCache(40), TurboKVCacheLite()]

    def load_model(model_id: str):
        assert model_id == "Qwen/Qwen3.5-0.8B"
        calls.append({"op": "load"})
        return model, object()

    def generate(**kwargs):
        calls.append({"op": "generate", **kwargs})

    def compact(cache):
        assert cache is turbo_cache
        assert [item["op"] for item in calls] == ["load", "generate", "generate"]
        calls.append({"op": "compact"})
        return 1

    metrics = run_paired_prefill(
        model_id="Qwen/Qwen3.5-0.8B",
        prompt_ids=[4, 5, 6],
        deps=PairedPrefillDeps(
            load_model=load_model,
            make_fp16_cache=lambda _: fp16_cache,
            make_turbo_cache=lambda _: turbo_cache,
            generate=generate,
            compact_turbo_cache=compact,
        ),
    )

    assert [item["op"] for item in calls] == ["load", "generate", "generate", "compact"]
    assert all(item["prompt"] == [4, 5, 6] for item in calls if item["op"] == "generate")
    assert all(item["max_tokens"] == 1 for item in calls if item["op"] == "generate")
    assert metrics["paired_prefill"] is True
    assert metrics["compaction_calls_reported"] == 1
    assert metrics["fp16_baseline_bytes"] == 2_000
    assert metrics["packed_state_bytes"] == 700
    assert metrics["byte_reduction"] == pytest.approx(0.6)


def test_paired_prefill_propagates_nonqualifying_contract() -> None:
    model = _Model()
    with pytest.raises(CompressionContractError, match="TurboKVCache"):
        run_paired_prefill(
            model_id="Qwen/Qwen3.5-0.8B",
            prompt_ids=[1],
            deps=PairedPrefillDeps(
                load_model=lambda _: (model, object()),
                make_fp16_cache=lambda _: [KVCache(40), KVCache(2_000)],
                make_turbo_cache=lambda _: [KVCache(40), KVCache(2_000)],
                generate=lambda **_: None,
                compact_turbo_cache=lambda _: 0,
            ),
        )
