"""llama.cpp backend — cross-platform (macOS, Linux, Windows, iOS, Android)."""

from __future__ import annotations
import logging
import time
from .base import BackendBase, BackendCapabilities, GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)


class LlamaCppBackend(BackendBase):
    capabilities = BackendCapabilities(
        name="llamacpp",
        primary="llamacpp",
        cuda=True,
        metal=True,
        supports_batching=True,
        supports_streaming=True,
        supports_turboquant=False,
        supports_spec_decode=False,
    )

    def __init__(self, model_path: str | None = None, n_ctx: int = 4096):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self._model = None
        self._load_error: str | None = None

    def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa

            return True
        except ImportError:
            return False

    def _load(self) -> None:
        if self._model is None and self.model_path:
            try:
                from llama_cpp import Llama

                self._model = Llama(model_path=self.model_path, n_ctx=self.n_ctx)
            except Exception as e:
                logger.warning(
                    "Failed to load llamacpp model from %s: %s", self.model_path, e
                )
                self._load_error = str(e)

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        self._load()
        if self._model is None:
            return GenerateResponse(
                text="",
                tokens=0,
                elapsed_ms=0,
                backend="llamacpp",
                metadata={"error": "gguf not loaded"},
            )
        t0 = time.time()
        out = self._model(
            req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stop=req.stop,
        )
        text = out["choices"][0]["text"] if isinstance(out, dict) else str(out)
        elapsed = int((time.time() - t0) * 1000)
        return GenerateResponse(
            text=text,
            tokens=out.get("usage", {}).get("completion_tokens", 1)
            if isinstance(out, dict)
            else len(text.split()),
            elapsed_ms=elapsed,
            backend="llamacpp",
        )
