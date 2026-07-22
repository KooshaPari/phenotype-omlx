"""TensorRT-LLM backend — NVIDIA high-throughput inference."""

from __future__ import annotations
import logging
import time
from .base import (
    BackendBase,
    BackendCapabilities,
    GenerateRequest,
    GenerateResponse,
    LazyBackendMixin,
)

logger = logging.getLogger(__name__)


class TensorrtBackend(LazyBackendMixin, BackendBase):
    capabilities = BackendCapabilities(
        name="tensorrt",
        primary="tensorrt",
        cuda=True,
        metal=False,
        supports_batching=True,
        supports_streaming=True,
        supports_turboquant=True,
        supports_spec_decode=True,
    )

    def __init__(self, engine_path: str | None = None):
        self.engine_path = engine_path
        self._runner = None
        self._load_error: str | None = None

    def is_available(self) -> bool:
        try:
            import tensorrt_llm  # noqa

            return True
        except ImportError:
            return False

    def _load_backend(self) -> None:
        if self._runner is None and self.engine_path:
            try:
                from tensorrt_llm.runtime import ModelRunner

                self._runner = ModelRunner.from_engine_file(self.engine_path)
            except Exception as e:
                logger.warning(
                    "Failed to load tensorrt model from %s: %s", self.engine_path, e
                )
                self._load_error = str(e)

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        self._ensure_loaded()
        if self._runner is None:
            return self._error_response(req, "engine not loaded", "tensorrt")
        t0 = time.time()
        out = self._runner.generate(req.prompt, max_new_tokens=req.max_tokens)
        text = out[0] if isinstance(out, list) else str(out)
        elapsed = int((time.time() - t0) * 1000)
        return GenerateResponse(
            text=text, tokens=len(text.split()), elapsed_ms=elapsed, backend="tensorrt"
        )
