"""Hybrid dispatch adapter — implements DispatchPort over existing backends."""

from __future__ import annotations

import asyncio
import logging
import time

from ..ports.inference import DispatchPort, InferenceRequest, InferenceResponse
from ..engines.hybrid_dispatch import (
    HybridDispatch,
    HybridConfig,
    DispatchPolicy,
)
from ..backends import GenerateRequest

logger = logging.getLogger(__name__)


class HybridDispatchAdapter(DispatchPort):
    """Adapter that bridges the existing HybridDispatch engine to DispatchPort.

    Routes requests through the multi-engine dispatcher and converts
    between port-level and backend-level dataclasses.
    """

    def __init__(self, config: HybridConfig | None = None) -> None:
        self._dispatch = HybridDispatch(config)

    def route(self, request: InferenceRequest) -> str:
        """Return the engine name that would handle this request."""
        available = self._dispatch.available()
        if not available:
            return "none"
        if DispatchPolicy.MLX in available:
            return "mlx"
        if DispatchPolicy.SGLANG in available:
            return "sglang"
        return str(available[0].value)

    async def dispatch(self, request: InferenceRequest) -> InferenceResponse:
        """Dispatch via the engine selector and convert the result."""
        backend_req = GenerateRequest(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop or [],
        )
        results = self._dispatch.generate(backend_req)
        if not results:
            return InferenceResponse(
                text="",
                tokens_generated=0,
                latency_ms=0.0,
                model="none",
            )
        best = results[0]
        return InferenceResponse(
            text=best.text,
            tokens_generated=best.tokens,
            latency_ms=float(best.elapsed_ms),
            model=best.backend,
        )
