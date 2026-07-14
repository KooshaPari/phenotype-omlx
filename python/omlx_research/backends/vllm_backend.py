"""vLLM backend — optional, NVIDIA primary."""

from __future__ import annotations
import time
from .base import BackendBase, BackendCapabilities, GenerateRequest, GenerateResponse


class VllmBackend(BackendBase):
    capabilities = BackendCapabilities(
        name="vllm",
        primary="vllm",
        cuda=True,
        metal=False,
        supports_batching=True,
        supports_streaming=True,
        supports_turboquant=False,
        supports_spec_decode=True,
    )

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._llm = None

    def is_available(self) -> bool:
        try:
            import vllm  # noqa
            return True
        except ImportError:
            return False

    def _load(self) -> None:
        if self._llm is None and self.model_path:
            from vllm import LLM, SamplingParams
            self._llm = LLM(model=self.model_path)
            self._sampling = SamplingParams

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        self._load()
        if self._llm is None:
            return GenerateResponse(text="", tokens=0, elapsed_ms=0, backend="vllm",
                                    metadata={"error": "no model"})
        t0 = time.time()
        sp = self._sampling(temperature=req.temperature, max_tokens=req.max_tokens)
        out = self._llm.generate([req.prompt], sp)
        text = out[0].outputs[0].text if out else ""
        elapsed = int((time.time() - t0) * 1000)
        return GenerateResponse(text=text, tokens=len(text.split()), elapsed_ms=elapsed, backend="vllm")
