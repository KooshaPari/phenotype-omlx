"""Inference ports — pure ABC contracts for text generation, cache, and dispatch.

These define the *domain* interfaces that adapters implement. Backends
already satisfy the ``InferencePort`` contract via duck typing; explicit
inheritance is optional and will be added incrementally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    stop: Optional[list[str]] = None


@dataclass
class InferenceResponse:
    text: str
    tokens_generated: int
    latency_ms: float
    model: str


class InferencePort(ABC):
    """Port for text generation inference."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> InferenceResponse: ...

    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs) -> InferenceResponse: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...


class CachePort(ABC):
    """Port for KV cache management."""

    @abstractmethod
    def insert(self, token_id: int, attention_weight: float) -> None: ...

    @abstractmethod
    def evict(self) -> list: ...


class DispatchPort(ABC):
    """Port for multi-engine dispatch."""

    @abstractmethod
    def route(self, request: InferenceRequest) -> str: ...

    @abstractmethod
    async def dispatch(self, request: InferenceRequest) -> InferenceResponse: ...
