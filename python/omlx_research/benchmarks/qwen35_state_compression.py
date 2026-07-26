"""Fail-closed FR-5 E3 measurements for Qwen3.5 runtime cache state.

Qwen3.5 can combine linear-recurrent and full-attention layers.  An E3
compression claim is therefore valid only for the actual post-prefill
TurboKV state of the full-attention subset, measured against a paired FP16
cache for that same subset.  This module deliberately accepts cache objects,
not precomputed byte counts, so callers cannot turn dummy vectors or scalars
into a release metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class CompressionContractError(ValueError):
    """The runtime state cannot support an honest FR-5 E3 claim."""


_TURBO_CACHE_TYPES = frozenset({"TurboKVCache", "TurboKVCacheLite"})
_PACKED_ATTRS = (
    "_packed_keys",
    "_packed_values",
    "_key_norms",
    "_value_norms",
    "_turbo_packed_keys",
    "_turbo_packed_values",
    "_turbo_key_norms",
    "_turbo_value_norms",
)


def _require_qwen35(model_id: str) -> None:
    normalized = model_id.lower()
    if "qwen3.5" not in normalized or "qwen2.5" in normalized:
        raise CompressionContractError(
            f"FR-5 E3 state metrics require Qwen3.5 only, got {model_id!r}"
        )


def _require_cache_object(cache: Any, *, role: str, index: int) -> None:
    if cache is None or isinstance(cache, (bool, bytes, bytearray, int, float, str)):
        raise CompressionContractError(
            f"{role} cache at layer {index} must be a runtime cache object, not scalar data"
        )


def _nbytes(value: Any) -> int:
    """Read bytes only from array-shaped runtime fields; never coerce scalars."""

    if value is None:
        return 0
    nbytes = getattr(value, "nbytes", None)
    if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes < 0:
        return 0
    return nbytes


def _cache_resident_bytes(cache: Any, *, role: str, index: int) -> int:
    _require_cache_object(cache, role=role, index=index)
    direct = _nbytes(cache)
    if direct > 0:
        return direct
    total = _nbytes(getattr(cache, "keys", None)) + _nbytes(getattr(cache, "values", None))
    inner = getattr(cache, "_kv", None) or getattr(cache, "_inner_kv", None)
    if inner is not None and inner is not cache:
        total += _cache_resident_bytes(inner, role=role, index=index)
    if total <= 0:
        raise CompressionContractError(
            f"{role} cache at layer {index} exposes no measurable runtime state"
        )
    return total


def _packed_state_bytes(cache: Any, *, index: int) -> int:
    _require_cache_object(cache, role="compacted", index=index)
    if type(cache).__name__ not in _TURBO_CACHE_TYPES:
        raise CompressionContractError(
            f"full-attention layer {index} is not a TurboKVCache runtime state"
        )
    compacted = bool(
        getattr(cache, "_compacted", False) or getattr(cache, "_is_compressed", False)
    )
    if not compacted:
        raise CompressionContractError(
            f"TurboKVCache at full-attention layer {index} is not compacted"
        )
    packed = sum(_nbytes(getattr(cache, attr, None)) for attr in _PACKED_ATTRS)
    if packed <= 0:
        raise CompressionContractError(
            f"TurboKVCache at full-attention layer {index} has no packed runtime state"
        )
    return packed


def _full_attention_indices(model: Any) -> list[int]:
    layers = getattr(model, "layers", None)
    if not isinstance(layers, Sequence) or not layers:
        raise CompressionContractError("model must expose non-empty runtime layers")
    indices = [index for index, layer in enumerate(layers) if not getattr(layer, "is_linear", False)]
    if not indices:
        raise CompressionContractError("Qwen3.5 model exposes no full-attention layers")
    return indices


def measure_qwen35_runtime_state_compression(
    *,
    model_id: str,
    model: Any,
    fp16_cache: Sequence[Any],
    compacted_cache: Sequence[Any],
) -> dict[str, Any]:
    """Measure a paired, post-prefill Qwen3.5 E3 compression contract.

    ``fp16_cache`` and ``compacted_cache`` must come from separate executions
    of the identical prompt through the same loaded model.  No byte count is
    accepted from a caller: every value is derived from cache object state.
    Raises :class:`CompressionContractError` instead of emitting an apparent
    zero-value measurement whenever the architecture or state is ineligible.
    """

    _require_qwen35(model_id)
    full_attention = _full_attention_indices(model)
    if len(fp16_cache) != len(compacted_cache):
        raise CompressionContractError("paired FP16 and compacted cache lists differ in length")
    if max(full_attention) >= len(fp16_cache):
        raise CompressionContractError("cache list does not cover every full-attention layer")

    baseline_bytes = 0
    resident_bytes = 0
    packed_bytes = 0
    for index in full_attention:
        baseline_bytes += _cache_resident_bytes(fp16_cache[index], role="FP16", index=index)
        resident_bytes += _cache_resident_bytes(
            compacted_cache[index], role="compacted", index=index
        )
        packed_bytes += _packed_state_bytes(compacted_cache[index], index=index)

    if baseline_bytes <= 0 or resident_bytes <= 0 or packed_bytes <= 0:
        raise CompressionContractError("runtime cache measurement contains nonpositive bytes")
    if resident_bytes >= baseline_bytes:
        raise CompressionContractError(
            "compacted full-attention runtime state did not reduce bytes versus paired FP16 cache"
        )

    reduction = 1.0 - (resident_bytes / baseline_bytes)
    return {
        "state_contract": "qwen35_full_attention_turbokv_post_prefill_v1",
        "model_id": model_id,
        "full_attention_layers": full_attention,
        "packed_state_bytes": packed_bytes,
        "fp16_baseline_bytes": baseline_bytes,
        "resident_state_bytes": resident_bytes,
        "byte_reduction": reduction,
        "e3_compression_qualifying": True,
    }
