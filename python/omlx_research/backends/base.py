"""Backend adapter base types."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)


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
    primary: str  # "mlx" | "metal" | "vllm" | "tensorrt" | "sglang" | "llamacpp" | "pt-mps" | "pt-cuda" | "pt-cpu"
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


class LazyBackendMixin:
    """Mixin providing lazy one-shot engine loading with error capture.

    Subclasses implement :meth:`_load_backend` which must set
    ``self._engine`` on success or leave it ``None`` on failure.
    ``_ensure_loaded`` is idempotent — it calls ``_load_backend``
    at most once and stores any exception in ``self._load_error``.

    Subclasses use ``self._engine`` after ``_ensure_loaded()`` returns;
    a ``None`` value means the backend could not be loaded.
    """

    _engine: Any = None
    _load_error: str | None = None
    _loaded: bool = False

    def _ensure_loaded(self) -> None:
        """Lazy-load the backend exactly once."""
        if self._loaded:
            return
        try:
            self._load_backend()
        except Exception as e:
            logger.warning("Backend load failed: %s", e)
            self._load_error = str(e)
        self._loaded = True

    def _load_backend(self) -> None:
        """Subclass hook: set ``self._engine`` on success.

        Must not raise — exceptions are caught by :meth:`_ensure_loaded`.
        """

    def _error_response(
        self, req: GenerateRequest, msg: str, backend: str = ""
    ) -> GenerateResponse:
        """Build an error :class:`GenerateResponse`."""
        return GenerateResponse(
            text="",
            tokens=0,
            elapsed_ms=0,
            backend=backend or getattr(self, "name", ""),
            metadata={"error": msg},
        )
