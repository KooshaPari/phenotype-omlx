"""Backend adapter base types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GenerateRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class GenerateResponse:
    text: str
    tokens: int
    elapsed_ms: int
    backend: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class BackendCapabilities:
    name: str
    primary: str          # "mlx" | "metal" | "vllm" | "tensorrt" | "sglang" | "llamacpp" | "pt-mps" | "pt-cuda" | "pt-cpu"
    cuda: bool = False
    metal: bool = False
    supports_batching: bool = True
    supports_streaming: bool = True
    supports_turboquant: bool = False
    supports_spec_decode: bool = False


class BackendBase(ABC):
    name: str = ""
    capabilities: BackendCapabilities

    @abstractmethod
    def generate(self, req: GenerateRequest) -> GenerateResponse: ...

    @abstractmethod
    def is_available(self) -> bool: ...
