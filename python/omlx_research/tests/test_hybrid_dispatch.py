"""Contract tests for fail-closed hybrid backend dispatch."""

from __future__ import annotations

import pytest

from omlx_research.backends import GenerateRequest
from omlx_research.engines.hybrid_dispatch import (
    DispatchPolicy,
    HybridDispatch,
    HybridConfig,
)


class _Backend:
    def __init__(self, name: str = "fake") -> None:
        self.name = name

    def is_available(self) -> bool:
        return True

    def generate(self, req: GenerateRequest) -> dict[str, str]:
        return {"backend": self.name, "prompt": req.prompt}


def _dispatch(backends: dict[DispatchPolicy, _Backend]) -> HybridDispatch:
    dispatch = HybridDispatch.__new__(HybridDispatch)
    dispatch.config = HybridConfig()
    dispatch._backends = backends
    return dispatch


def test_generate_rejects_unavailable_requested_backend() -> None:
    dispatch = _dispatch({DispatchPolicy.SGLANG: _Backend("sglang")})
    with pytest.raises(Exception, match="vllm"):
        dispatch.generate(GenerateRequest("hello"), DispatchPolicy.VLLM)


def test_auto_dispatch_does_not_choose_unavailable_backend() -> None:
    dispatch = _dispatch({DispatchPolicy.SGLANG: _Backend("sglang")})
    dispatch._auto_pick = lambda: DispatchPolicy.SGLANG
    assert dispatch.generate(GenerateRequest("hello")) == [
        {"backend": "sglang", "prompt": "hello"}
    ]


def test_fanout_rejects_when_no_backend_is_available() -> None:
    dispatch = _dispatch({})
    with pytest.raises(Exception, match="no available backends"):
        dispatch.generate(GenerateRequest("hello"), DispatchPolicy.FANOUT)
