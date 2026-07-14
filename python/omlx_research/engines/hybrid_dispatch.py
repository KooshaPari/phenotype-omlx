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
            return [b.generate(req) for _, b in self._backends.items() if b.is_available()]
        b = self._backends.get(pol)
        if b is None:
            return []
        return [b.generate(req)]

    def _auto_pick(self) -> DispatchPolicy:
        # Prefer Metal/MLX on Apple Silicon, NVIDIA engines otherwise.
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                return DispatchPolicy.METAL
        except ImportError:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                return DispatchPolicy.SGLANG
        except ImportError:
            pass
        return DispatchPolicy.MLX
