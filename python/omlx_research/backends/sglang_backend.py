"""SGLang backend — primary inference engine for NVIDIA clusters."""

from __future__ import annotations
import logging
import time
from .base import BackendBase, BackendCapabilities, GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)


class SglangBackend(BackendBase):
    capabilities = BackendCapabilities(
        name="sglang",
        primary="sglang",
        cuda=True,
        metal=False,
        supports_batching=True,
        supports_streaming=True,
        supports_turboquant=True,
        supports_spec_decode=True,
    )

    def __init__(
        self, model_path: str | None = None, base_url: str = "http://127.0.0.1:30000"
    ):
        self.model_path = model_path
        self.base_url = base_url
        self._runtime = None
        self._load_error: str | None = None

    def is_available(self) -> bool:
        try:
            import sglang  # noqa

            return True
        except ImportError:
            return False

    def _load(self) -> None:
        if self._runtime is None:
            try:
                import sglang

                if self.model_path:
                    self._runtime = sglang.Runtime(
                        model_path=self.model_path, base_url=self.base_url
                    )
            except Exception as e:
                logger.warning(
                    "Failed to load sglang model from %s: %s", self.model_path, e
                )
                self._load_error = str(e)

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        self._load()
        if self._runtime is None:
            return GenerateResponse(
                text="",
                tokens=0,
                elapsed_ms=0,
                backend="sglang",
                metadata={"error": "runtime not loaded"},
            )
        t0 = time.time()
        out = self._runtime.generate(req.prompt, max_new_tokens=req.max_tokens)
        text = out if isinstance(out, str) else str(out)
        elapsed = int((time.time() - t0) * 1000)
        return GenerateResponse(
            text=text, tokens=len(text.split()), elapsed_ms=elapsed, backend="sglang"
        )
