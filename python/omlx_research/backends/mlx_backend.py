"""MLX backend — primary path on Apple Silicon."""

from __future__ import annotations
import time
import os

from .base import BackendBase, BackendCapabilities, GenerateRequest, GenerateResponse


class MlxBackend(BackendBase):
    capabilities = BackendCapabilities(
        name="mlx",
        primary="mlx",
        cuda=False,
        metal=True,
        supports_batching=True,
        supports_streaming=True,
        supports_turboquant=True,
        supports_spec_decode=True,
    )

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._perf_module = None  # lazy-loaded pyo3 Rust extension

    def is_available(self) -> bool:
        try:
            import mlx.core  # noqa
            return True
        except ImportError:
            return False

    def _rust_perf(self):
        """Lazy-import the pyo3 `_perf` extension built from perf-core/.
        Returns the module or None if not built (caller falls back to Python)."""
        if self._perf_module is not None:
            return self._perf_module
        try:
            import _perf  # maturin develop installs this top-level
            self._perf_module = _perf
            return _perf
        except ImportError:
            self._perf_module = False  # sentinel — don't retry
            return None

    def turbo_quant_encode_array(
        self, data, group_size: int = 64, bits: int = 4,
    ) -> dict | None:
        """Encode `data` (array-like of f32) into a TurboQuant 4-bit packing.

        Uses the Rust SIMD implementation from perf-core/turbo-quant when the
        pyo3 extension is built; otherwise returns None (caller falls back).
        """
        perf = self._rust_perf()
        if perf is None:
            return None
        return perf.turbo_quant_encode(list(map(float, data)), group_size, bits)

    def turbo_quant_decode_array(
        self, packed, scales, zeros, n: int, group_size: int = 64, bits: int = 4,
    ) -> list | None:
        """Inverse of turbo_quant_encode_array (Rust SIMD path)."""
        perf = self._rust_perf()
        if perf is None:
            return None
        return list(perf.turbo_quant_decode(packed, scales, zeros, n, group_size, bits))

    def _load(self) -> None:
        if self._model is None and self.model_path:
            import mlx_lm
            self._model, self._tokenizer = mlx_lm.load(self.model_path)

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        self._load()
        if self._model is None:
            return GenerateResponse(text="", tokens=0, elapsed_ms=0, backend="mlx",
                                    metadata={"error": "no model"})
        import mlx_lm
        t0 = time.time()
        text = mlx_lm.generate(
            self._model,
            self._tokenizer,
            req.prompt,
            max_tokens=req.max_tokens,
            verbose=False,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        return GenerateResponse(text=text, tokens=len(text.split()), elapsed_ms=elapsed_ms, backend="mlx")
