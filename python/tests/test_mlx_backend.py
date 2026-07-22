"""Contract tests for the MLX backend's canonical TurboQuant bridge."""

from __future__ import annotations

from omlx_research.backends.mlx_backend import MlxBackend


def test_turbo_quant_decode_consumes_encoder_payload(monkeypatch) -> None:
    backend = MlxBackend()
    calls: list[tuple] = []

    class PerfModule:
        @staticmethod
        def turbo_quant_encode(values, group_size, bits):
            return {
                "shape": [len(values)],
                "bits": bits,
                "group_size": group_size,
                "packed": b"\x21",
                "scales": [0.5],
                "zeros": [-1.0],
            }

        @staticmethod
        def turbo_quant_decode(shape, bits, group_size, packed, scales, zeros):
            calls.append((shape, bits, group_size, packed, scales, zeros))
            return [-0.5, 0.0]

    monkeypatch.setattr(backend, "_rust_perf", lambda: PerfModule)

    payload = backend.turbo_quant_encode_array([-0.5, 0.0], group_size=32, bits=4)
    decoded = backend.turbo_quant_decode_array(payload)

    assert decoded == [-0.5, 0.0]
    assert calls == [([2], 4, 32, b"\x21", [0.5], [-1.0])]
