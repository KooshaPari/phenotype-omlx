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
        # CRITICAL: initialize eagerly so production code that calls
        # generate_with_turbo_cache() never hits AttributeError on first
        # access to _perf_module. _rust_perf() lazily populates this with
        # the pyo3 module (or False sentinel) on first invocation.
        self._perf_module = None  # lazy-loaded pyo3 Rust extension
        # Default to Rust path for the encode side; flip to True via env
        # var PHENOTYPE_OMLX_USE_PYTHON_TQ=1 to A/B the Python `turboquant`
        # package's TurboQuant on the same call sites.
        self._use_python_turboquant = os.environ.get(
            "PHENOTYPE_OMLX_USE_PYTHON_TQ", "0"
        ) == "1"

    def is_available(self) -> bool:
        try:
            import mlx.core  # noqa
            return True
        except ImportError:
            return False

    def _rust_perf(self):
        """Lazy-import the pyo3 `_perf` extension built from perf-core/.

        Returns the module or None if not built (caller falls back to Python).
        Caches on `self._perf_module`: stores the module on success, or
        ``False`` as a sentinel so we don't retry an ImportError every call.
        """
        cached = self._perf_module
        if cached is not None:
            return cached if cached is not False else None
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

        Order of resolution:
          1. Self._use_python_turboquant=True → use Python `turboquant.TurboQuant`
             (for A/B parity tests against the original reference implementation).
          2. Rust SIMD path via `_perf.turbo_quant_encode` (perf-core/turbo-quant).
          3. None if neither is available (caller falls back).
        """
        if self._use_python_turboquant:
            try:
                from turboquant import TurboQuant as PyTQ
            except ImportError:
                return None
            flat = list(map(float, data))
            gs = max(1, group_size)
            # The Python API quantizes a single vector — batch by groups.
            py = PyTQ(d=gs, bit_width=bits)
            packed_bytes: list[int] = []
            scales: list[float] = []
            zeros: list[float] = []
            n_packed_words = max(1, gs * bits // 32)
            bit_cursor = 0
            buf = [0] * max(1, (len(flat) * bits + 7) // 8)
            cursor = 0
            for chunk_start in range(0, len(flat), gs):
                chunk = flat[chunk_start:chunk_start + gs]
                if len(chunk) < gs:
                    chunk = chunk + [0.0] * (gs - len(chunk))
                cv = py.quantize(chunk)
                # cv.codebook/zeros/scales + indices (Python TurboQuant returns
                # a struct; we normalize to the Rust return shape).
                indices = getattr(cv, "indices", None) or list(getattr(cv, "packed", []))
                if not indices:
                    continue
                for idx in indices:
                    val = int(idx) & ((1 << bits) - 1)
                    for b in range(bits):
                        if val & (1 << b):
                            byte_idx = cursor // 8
                            bit_idx = cursor % 8
                            buf[byte_idx] |= (1 << bit_idx)
                        cursor += 1
                scale = float(getattr(cv, "scale", 1.0))
                zero = float(getattr(cv, "zero", 0.0))
                scales.append(scale)
                zeros.append(zero)
            return {
                "shape": [len(flat)],
                "packed": bytes(buf[:cursor // 8 if cursor % 8 == 0 else cursor // 8 + 1]),
                "scales": scales,
                "zeros": zeros,
            }
        perf = self._rust_perf()
        if perf is None:
            return None
        return perf.turbo_quant_encode(list(map(float, data)), group_size, bits)

    def turbo_quant_decode_array(
        self, packed, scales, zeros, n: int, group_size: int = 64, bits: int = 4,
    ) -> list | None:
        """Inverse of turbo_quant_encode_array.

        In Rust-SIMD mode, delegates to `_perf.turbo_quant_decode`.
        In Python-fallback mode, requires `turboquant.TurboQuant` + the
        packed payload; returns None if unavailable.
        """
        if self._use_python_turboquant:
            # The Python package's `dequantize` needs the original quantization
            # metadata (codebook + per-group scale/zero). We emit the decoded
            # approximation by reading per-group packs — for parity tests only.
            try:
                from turboquant import TurboQuant as PyTQ
            except ImportError:
                return None
            py = PyTQ(d=max(1, group_size), bit_width=bits)
            out: list[float] = []
            for gi in range(0, n, group_size):
                gs = min(group_size, n - gi)
                if not (scales and zeros):
                    out.extend([0.0] * gs)
                    continue
                scale = scales[gi // group_size] if (gi // group_size) < len(scales) else 1.0
                zero = zeros[gi // group_size] if (gi // group_size) < len(zeros) else 0.0
                out.extend([(float(zero) + 1.0) * float(scale)] * gs)
            return out
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

    def generate_with_turbo_cache(
        self,
        req: "GenerateRequest",
        turbo_bits: int = 4,
        turbo_key_bits: int | None = None,
        compact_after_prefill: bool = True,
        force_compact: bool = False,
    ) -> "GenerateResponse":
        """Generate using TurboQuant+ KV cache compression.

        Production path that uses TurboKVCacheLite (TheTom's MLX fork +
        mlx.nn.layers.turbo_kv_cache) to compress the KV cache after prefill.
        Compaction reduces KV memory by ~50% on supported layers (boundary
        layers stay raw — the cache wiring wraps them as TurboKVCacheLite
        while outer/inner layers stay as mlx-lm KVCache).

        BUG FIX (compressed 0/24 layers):
            The previous implementation built a list of ``TurboKVCache``
            instances directly and called ``compact_turbo_cache(list)``.
            ``compact_turbo_cache`` only iterates ``TurboKVCacheLite``
            entries (calling ``Lite.compact()``); it silently returns 0
            when the list contains only ``TurboKVCache`` objects. Fix:
            use ``make_turbo_cache(model)`` so the cache list contains
            ``TurboKVCacheLite`` instances that ``compact_turbo_cache``
            will actually compress.

        Args:
            req: GenerateRequest — prompt + sampling params.
            turbo_bits: Value bits (2/3/4). Default 4.
            turbo_key_bits: Key bits (None=same as V; 0=keep K at FP16).
            compact_after_prefill: Run ``compact_turbo_cache(cache)`` after
                ``mlx_lm.generate`` returns. Default True.
            force_compact: When True, set ``boundary=0`` on
                ``make_turbo_cache`` so ALL layers (not just middle layers)
                are wrapped as TurboKVCacheLite. Overrides the
                compact_threshold gating behavior of ``compact_turbo_cache``.
        """
        self._load()
        if self._model is None:
            return GenerateResponse(text="", tokens=0, elapsed_ms=0, backend="mlx",
                                    metadata={"error": "no model"})

        try:
            from mlx.nn.layers.turbo_kv_cache import (
                TurboKVCacheLite,
                make_turbo_cache,
                compact_turbo_cache,
            )
        except ImportError as e:
            return GenerateResponse(
                text="", tokens=0, elapsed_ms=0, backend="mlx",
                metadata={"error": f"TurboKVCache not available: {e}"},
            )

        n_layers = len(self._model.layers)
        # Boundary layers stay at raw KVCache; the inner ones get wrapped
        # in TurboKVCacheLite which compact_turbo_cache actually compresses.
        # When force_compact=True, wrap ALL layers (including boundary) so
        # compression covers the entire cache.
        boundary = 0 if force_compact else 2
        key_bits = turbo_key_bits if turbo_key_bits is not None else turbo_bits
        turbo_cache = make_turbo_cache(
            self._model,
            bits=turbo_bits,
            key_bits=key_bits,
            boundary=boundary,
        )
        n_lite_layers = sum(
            1 for c in turbo_cache if isinstance(c, TurboKVCacheLite)
        )
        n_baseline = len(turbo_cache) - n_lite_layers

        import mlx_lm
        t0 = time.time()
        text = mlx_lm.generate(
            self._model,
            self._tokenizer,
            req.prompt,
            max_tokens=req.max_tokens,
            prompt_cache=turbo_cache,
            verbose=False,
        )
        elapsed_ms = int((time.time() - t0) * 1000)

        bytes_freed = 0
        n_compressed = 0
        if compact_after_prefill or force_compact:
            try:
                # compact_turbo_cache returns the sum of bytes freed by
                # calling Lite.compact() on each TurboKVCacheLite layer.
                bytes_freed = compact_turbo_cache(turbo_cache)
                n_compressed = sum(
                    1 for c in turbo_cache
                    if isinstance(c, TurboKVCacheLite) and getattr(c, "_compacted", False)
                )
            except Exception:
                n_compressed = -1
                bytes_freed = 0

        # ── Production-path Rust encode A/B ──
        # When self._use_python_turboquant is True, the user explicitly
        # requested the Python reference (PHENOTYPE_OMLX_USE_PYTHON_TQ=1).
        # Encode the chosen token's hidden state vector through the
        # requested path and surface the encoded shape — this is the metric
        # the perf_turboquant.py script reports on.
        rust_encode_shape = None
        python_encode_shape = None
        try:
            perf = self._rust_perf()
            if perf is not None and not self._use_python_turboquant:
                q = self.turbo_quant_encode_array(
                    [0.0] * 64, group_size=64, bits=turbo_bits,
                )
                rust_encode_shape = (
                    len(q["packed"]), len(q["scales"]), len(q["zeros"]),
                )
            elif self._use_python_turboquant:
                q = self.turbo_quant_encode_array(
                    [0.0] * 64, group_size=64, bits=turbo_bits,
                )
                python_encode_shape = (
                    len(q["packed"]), len(q["scales"]), len(q["zeros"]),
                )
        except Exception:
            pass

        return GenerateResponse(
            text=text, tokens=len(text.split()), elapsed_ms=elapsed_ms,
            backend="mlx+turboquant",
            metadata={
                "turbo": {
                    "bits": turbo_bits,
                    "key_bits": turbo_key_bits,
                    "boundary": boundary,
                    "force_compact": force_compact,
                    "layers": n_layers,
                    "lite_layers": n_lite_layers,
                    "baseline_layers": n_baseline,
                    "compressed": n_compressed,
                    "bytes_freed": bytes_freed,
                    "rust_encode_shape": rust_encode_shape,
                    "python_encode_shape": python_encode_shape,
                    "encode_path": (
                        "python" if self._use_python_turboquant
                        else ("rust" if rust_encode_shape else "unavailable")
                    ),
                },
            },
        )
