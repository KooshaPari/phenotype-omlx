"""Contract tests for Qwen3.5 FR-5 E3 runtime-state compression evidence."""

from __future__ import annotations

import pytest

from omlx_research.benchmarks.qwen35_state_compression import (
    CompressionContractError,
    measure_qwen35_runtime_state_compression,
)


class _Array:
    """MLX-array-shaped test double; inputs remain cache objects, never byte scalars."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes


class _FullAttentionLayer:
    is_linear = False


class _LinearLayer:
    is_linear = True


class _Model:
    layers = [_LinearLayer(), _FullAttentionLayer(), _FullAttentionLayer()]


class KVCache:
    def __init__(self, nbytes: int) -> None:
        self.keys = _Array(nbytes // 2)
        self.values = _Array(nbytes // 2)


class TurboKVCacheLite:
    def __init__(self, *, packed: int, norms: int, resident: int) -> None:
        self._compacted = True
        self._turbo_packed_keys = _Array(packed // 2)
        self._turbo_packed_values = _Array(packed // 2)
        self._turbo_key_norms = _Array(norms // 2)
        self._turbo_value_norms = _Array(norms // 2)
        self._resident = resident

    @property
    def nbytes(self) -> int:
        return self._resident


def test_measures_only_real_full_attention_cache_state() -> None:
    """E3 values derive from paired post-prefill cache objects, never input scalars."""

    baseline = [KVCache(80), KVCache(1_000), KVCache(2_000)]
    compressed = [KVCache(80), TurboKVCacheLite(packed=400, norms=80, resident=600),
                  TurboKVCacheLite(packed=800, norms=120, resident=1_100)]

    metrics = measure_qwen35_runtime_state_compression(
        model_id="mlx-community/Qwen3.5-0.8B-OptiQ-4bit",
        model=_Model(),
        fp16_cache=baseline,
        compacted_cache=compressed,
    )

    assert metrics["state_contract"] == "qwen35_full_attention_turbokv_post_prefill_v1"
    assert metrics["full_attention_layers"] == [1, 2]
    assert metrics["packed_state_bytes"] == 1_400
    assert metrics["fp16_baseline_bytes"] == 3_000
    assert metrics["resident_state_bytes"] == 1_700
    assert metrics["byte_reduction"] == pytest.approx(1 - 1_700 / 3_000)
    assert metrics["e3_compression_qualifying"] is True


def test_rejects_qwen2_and_scalar_or_missing_cache_contracts() -> None:
    """No synthetic/dummy bytes can become an E3 compression claim."""

    with pytest.raises(CompressionContractError, match="Qwen3.5"):
        measure_qwen35_runtime_state_compression(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            model=_Model(),
            fp16_cache=[1, 2, 3],
            compacted_cache=[4, 5, 6],
        )

    with pytest.raises(CompressionContractError, match="cache object"):
        measure_qwen35_runtime_state_compression(
            model_id="Qwen/Qwen3.5-0.8B",
            model=_Model(),
            fp16_cache=[1, 2, 3],
            compacted_cache=[4, 5, 6],
        )


def test_fails_closed_without_compacted_full_attention_state() -> None:
    """A linear-only or uncompressed route is N/A, never a zero-byte E3 pass."""

    with pytest.raises(CompressionContractError, match="TurboKVCache"):
        measure_qwen35_runtime_state_compression(
            model_id="Qwen/Qwen3.5-0.8B",
            model=_Model(),
            fp16_cache=[KVCache(80), KVCache(1_000), KVCache(2_000)],
            compacted_cache=[KVCache(80), KVCache(1_000), KVCache(2_000)],
        )
