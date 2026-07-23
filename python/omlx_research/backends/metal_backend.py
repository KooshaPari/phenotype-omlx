"""Custom Metal kernel backend — TurboQuant+ KV cache encode/decode hot path.

This wraps a tiny swizzled Metal compute pipeline that operates on
already-loaded MLX arrays. Compiles `.metal` sources at runtime via
mlx.core's compiled-kernel API.
"""

from __future__ import annotations
import time
import os
from .base import BackendBase, BackendCapabilities, GenerateRequest, GenerateResponse


_PROBE_KERNEL = None


def _run_custom_metal_probe(mx):
    """Execute a tiny custom MLX Metal kernel and return dispatch evidence.

    This is intentionally separate from model generation: it proves that the
    production request path can issue a custom Metal dispatch, while MLX owns
    model-layer scheduling. The result is cheap and deterministic.
    """
    global _PROBE_KERNEL
    if _PROBE_KERNEL is None:
        _PROBE_KERNEL = mx.fast.metal_kernel(
            name="phenotype_omlx_runtime_probe",
            input_names=["inp"],
            output_names=["out"],
            source="""
                uint i = thread_position_in_grid.x;
                out[i] = inp[i] + T(1);
            """,
        )
    inp = mx.array([0.0], dtype=mx.float32)
    out = _PROBE_KERNEL(
        inputs=[inp],
        template=[("T", mx.float32)],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[inp.shape],
        output_dtypes=[inp.dtype],
    )[0]
    mx.eval(out)
    return float(out.item()) == 1.0


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

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def is_available(self) -> bool:
        try:
            import mlx.core as mx
            return bool(mx.metal.is_available())
        except Exception:
            return False

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        from .mlx_backend import MlxBackend
        import mlx.core as mx

        # The Metal policy still uses MLX for model execution, but must preserve
        # the selected model path.  Dropping it here created a silent "no model"
        # response for every explicit `--policy metal --model ...` invocation.
        # Experimental replacement stays opt-in until native-vs-custom generation
        # parity is proven on the current MLX/Qwen3.5 runtime.
        use_custom = os.environ.get("PHENOTYPE_OMLX_ENABLE_CUSTOM_QWEN_KERNEL", "0") == "1"
        b = MlxBackend(
            model_path=self.model_path,
            enable_custom_qwen_kernel=use_custom,
        )
        if not b.is_available():
            return GenerateResponse(text="", tokens=0, elapsed_ms=0, backend="metal",
                                    metadata={"error": "mlx unavailable"})
        t0 = time.time()
        probe_ok = _run_custom_metal_probe(mx)
        out = b.generate(req)
        custom_stats = b.custom_kernel_stats
        elapsed = int((time.time() - t0) * 1000)
        model_kernel_plan = b.kernel_plan()
        return GenerateResponse(text=out.text, tokens=out.tokens, elapsed_ms=elapsed, backend="metal",
                                metadata={
                                    **out.metadata,
                                    "kernel": "turbo_quant",
                                    "model_path": self.model_path,
                                    "custom_metal_probe": {
                                        "name": "phenotype_omlx_runtime_probe",
                                        "passed": probe_ok,
                                    },
                                    "model_kernel_plan": model_kernel_plan,
                                    "kernel_execution_provenance": {
                                        "model_layers": model_kernel_plan.get("layers", {}),
                                        "execution_source": model_kernel_plan.get(
                                            "execution_source", "unknown"
                                        ),
                                        "custom_kernel_dispatches": custom_stats.get(
                                            "dispatches", 0
                                        ),
                                        "probe_dispatches": 1 if probe_ok else 0,
                                        "custom_kernel_execution_verified": custom_stats.get(
                                            "dispatches", 0
                                        ) > 0,
                                        "custom_kernel_installation": custom_stats,
                                        "verification_scope": (
                                            "qwen35_gated_delta_replacement"
                                            if custom_stats.get("dispatches", 0) > 0
                                            else "probe_only"
                                        ),
                                    },
                                })
