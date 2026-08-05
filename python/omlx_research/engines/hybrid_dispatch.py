"""Hybrid dispatch — auto-pick engine (MLX, Metal, vLLM, SGLang, etc.)."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from ..backends import (
    BackendBase,
    MlxBackend,
    MetalKernelBackend,
    VllmBackend,
    TensorrtBackend,
    SglangBackend,
    LlamaCppBackend,
    GenerateRequest,
)


class DispatchPolicy(str, Enum):
    AUTO = "auto"
    MLX = "mlx"
    METAL = "metal"
    VLLM = "vllm"
    TENSORRT = "tensorrt"
    SGLANG = "sglang"
    LLAMACPP = "llamacpp"
    FANOUT = "fanout"
    SPEC_DECODE = "spec_decode"
    TIDAR = "tidar"


@dataclass
class HybridConfig:
    primary: DispatchPolicy = DispatchPolicy.AUTO
    fanout_engines: tuple[DispatchPolicy, ...] = (
        DispatchPolicy.MLX,
        DispatchPolicy.SGLANG,
        DispatchPolicy.VLLM,
        DispatchPolicy.LLAMACPP,
    )


class HybridDispatchError(RuntimeError):
    """Raised when a requested dispatch route has no usable backend."""

    def __init__(self, policy: DispatchPolicy, available: list[DispatchPolicy]) -> None:
        self.policy = policy
        self.available = available
        available_names = ", ".join(item.value for item in available) or "none"
        if not available:
            message = f"backend {policy.value!r} unavailable; no available backends"
        else:
            message = f"backend {policy.value!r} unavailable; available backends: {available_names}"
        super().__init__(message)


class HybridDispatch:
    def __init__(self, config: HybridConfig | None = None):
        self.config = config or HybridConfig()
        self._backends: dict[DispatchPolicy, BackendBase] = {}
        self._init_backends()

    def _init_backends(self) -> None:
        for name, cls in (
            (DispatchPolicy.MLX, MlxBackend),
            (DispatchPolicy.METAL, MetalKernelBackend),
            (DispatchPolicy.VLLM, VllmBackend),
            (DispatchPolicy.TENSORRT, TensorrtBackend),
            (DispatchPolicy.SGLANG, SglangBackend),
            (DispatchPolicy.LLAMACPP, LlamaCppBackend),
        ):
            b = cls()
            if b.is_available():
                self._backends[name] = b

    def available(self) -> list[DispatchPolicy]:
        return list(self._backends.keys())

    def generate(self, req: GenerateRequest, policy: DispatchPolicy | None = None) -> list:
        pol = policy or self.config.primary
        if pol == DispatchPolicy.AUTO:
            pol = self._auto_pick()
        if pol == DispatchPolicy.FANOUT:
            available = [(name, backend) for name, backend in self._backends.items() if backend.is_available()]
            if not available:
                raise HybridDispatchError(DispatchPolicy.FANOUT, self.available())
            return [backend.generate(req) for _, backend in available]
        b = self._backends.get(pol)
        if b is None or not b.is_available():
            raise HybridDispatchError(pol, self.available())
        return [b.generate(req)]

    def _auto_pick(self) -> DispatchPolicy:
        # Prefer Metal/MLX on Apple Silicon, NVIDIA engines otherwise.
        preferred: list[DispatchPolicy]
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                preferred = [DispatchPolicy.METAL, DispatchPolicy.MLX]
            else:
                preferred = []
        except ImportError:
            preferred = []
        if not preferred:
            try:
                import torch
                if torch.cuda.is_available():
                    preferred = [DispatchPolicy.SGLANG, DispatchPolicy.VLLM, DispatchPolicy.TENSORRT]
            except ImportError:
                pass
        preferred.extend([DispatchPolicy.MLX, DispatchPolicy.LLAMACPP])
        for candidate in preferred:
            backend = self._backends.get(candidate)
            if backend is not None and backend.is_available():
                return candidate
        raise HybridDispatchError(DispatchPolicy.AUTO, self.available())
