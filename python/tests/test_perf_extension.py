"""Contract tests for the packaged Rust TurboQuant extension."""

from __future__ import annotations

import math

import pytest

from omlx_research import _perf
from omlx_research.backends.mlx_backend import MlxBackend


def test_turbo_quant_round_trip_preserves_metadata_and_uses_bytes() -> None:
    values = [math.sin(index / 7.0) for index in range(129)]

    encoded = _perf.turbo_quant_encode(values, group_size=64, bits=4)

    assert encoded["shape"] == [len(values)]
    assert encoded["bits"] == 4
    assert encoded["group_size"] == 64
    assert isinstance(encoded["packed"], bytes)
    assert encoded["scales"]
    assert encoded["zeros"]

    decoded = _perf.turbo_quant_decode(
        encoded["shape"],
        encoded["bits"],
        encoded["group_size"],
        encoded["packed"],
        encoded["scales"],
        encoded["zeros"],
    )

    assert len(decoded) == len(values)
    assert max(abs(expected - actual) for expected, actual in zip(values, decoded)) < 0.25


@pytest.mark.parametrize("bits", [0, 1, 9])
def test_turbo_quant_rejects_unsupported_bit_width(bits: int) -> None:
    with pytest.raises(ValueError):
        _perf.turbo_quant_label_for_bits(bits)


@pytest.mark.parametrize("bits", [0, 1, 5, 8, 9])
def test_turbo_quant_public_bit_width_contract_is_consistent(bits: int) -> None:
    with pytest.raises(ValueError):
        _perf.turbo_quant_label_for_bits(bits)

    with pytest.raises(ValueError):
        _perf.turbo_quant_encode([0.0, 1.0], group_size=2, bits=bits)


@pytest.mark.parametrize(
    ("values", "group_size"),
    [([0.0], 0), ([float("nan")], 64), ([float("inf")], 64)],
)
def test_turbo_quant_encode_rejects_invalid_input(
    values: list[float], group_size: int
) -> None:
    with pytest.raises(ValueError):
        _perf.turbo_quant_encode(values, group_size=group_size, bits=4)


def test_turbo_quant_decode_rejects_malformed_payload() -> None:
    encoded = _perf.turbo_quant_encode([0.1] * 64, group_size=64, bits=4)

    with pytest.raises(ValueError):
        _perf.turbo_quant_decode(
            encoded["shape"],
            encoded["bits"],
            encoded["group_size"],
            encoded["packed"][:-1],
            encoded["scales"],
            encoded["zeros"],
        )


def test_turbo_quant_decode_rejects_oversized_output_before_allocation() -> None:
    max_elements = 1_048_576
    count = max_elements + 1

    with pytest.raises(ValueError, match="maximum"):
        _perf.turbo_quant_decode(
            [count],
            4,
            count,
            bytes((count * 4 + 7) // 8),
            [1.0],
            [0.0],
        )


def test_turbo_quant_encode_rejects_oversized_input_before_quantization() -> None:
    max_elements = 1_048_576

    with pytest.raises(ValueError, match="maximum"):
        _perf.turbo_quant_encode([0.0] * (max_elements + 1), group_size=64, bits=4)
