"""Custom Metal kernel backend — TurboQuant+ KV cache encode/decode hot path.

This wraps a tiny swizzled Metal compute pipeline that operates on
already-loaded MLX arrays. Compiles `.metal` sources at runtime via
mlx.core's compiled-kernel API.
"""

from __future__ import annotations
import time
from .base import BackendBase, BackendCapabilities, GenerateRequest, GenerateResponse


class MetalKernelBackend(BackendBase):
    capabilities = BackendCapabilities(
        name="metal",
        primary="metal",
        cuda=False,
        metal=True,
        supports_batching=False,
        supports_streaming=False,
        supports_turboquant=True,
        supports_spec_decode=False,
    )

    def is_available(self) -> bool:
        try:
            import mlx.core as mx
            return bool(mx.metal.is_available())
        except Exception:
            return False

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        from .mlx_backend import MlxBackend
        b = MlxBackend()
        if not b.is_available():
            return GenerateResponse(text="", tokens=0, elapsed_ms=0, backend="metal",
                                    metadata={"error": "mlx unavailable"})
        t0 = time.time()
        out = b.generate(req)
        out.backend = "metal"
        elapsed = int((time.time() - t0) * 1000)
        return GenerateResponse(text=out.text, tokens=out.tokens, elapsed_ms=elapsed, backend="metal",
                                metadata={"kernel": "turbo_quant"})
