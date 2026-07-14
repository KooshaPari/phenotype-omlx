"""Adapter layer: vLLM / TensorRT / SGLang / llama.cpp / MLX / Metal.

Each backend exposes a uniform `generate(prompt, **kw) -> str` interface that
the OMLX server / CLI / GUI can call without committing to a specific engine.
"""

from .base import BackendBase, BackendCapabilities, GenerateRequest, GenerateResponse
from .mlx_backend import MlxBackend
from .metal_backend import MetalKernelBackend
from .vllm_backend import VllmBackend
from .tensorrt_backend import TensorrtBackend
from .sglang_backend import SglangBackend
from .llamacpp_backend import LlamaCppBackend

__all__ = [
    "BackendBase",
    "BackendCapabilities",
    "GenerateRequest",
    "GenerateResponse",
    "MlxBackend",
    "MetalKernelBackend",
    "VllmBackend",
    "TensorrtBackend",
    "SglangBackend",
    "LlamaCppBackend",
]
