"""Uniform contract tests for every application-layer backend adapter.

These tests deliberately use unconfigured adapters.  Optional inference
engines are not required for the contract: an adapter must be safe to probe
and must return a typed response when no model/engine is configured.
"""

from __future__ import annotations

import pytest

from omlx_research.backends import (
    GenerateRequest,
    GenerateResponse,
    LlamaCppBackend,
    MetalKernelBackend,
    MlxBackend,
    SglangBackend,
    TensorrtBackend,
    VllmBackend,
)
from omlx_research.backends.base import BackendBase


BACKEND_FACTORIES = [
    pytest.param(MlxBackend, "mlx", id="mlx"),
    pytest.param(MetalKernelBackend, "metal", id="metal"),
    pytest.param(VllmBackend, "vllm", id="vllm"),
    pytest.param(TensorrtBackend, "tensorrt", id="tensorrt"),
    pytest.param(SglangBackend, "sglang", id="sglang"),
    pytest.param(LlamaCppBackend, "llamacpp", id="llamacpp"),
]


@pytest.mark.parametrize(("factory", "name"), BACKEND_FACTORIES)
def test_backend_capabilities_and_probe_are_uniform(factory, name: str) -> None:
    backend = factory()

    assert isinstance(backend, BackendBase)
    assert backend.capabilities.name == name
    assert backend.capabilities.primary == name
    assert isinstance(backend.is_available(), bool)


@pytest.mark.parametrize(("factory", "name"), BACKEND_FACTORIES)
def test_unconfigured_backend_returns_typed_response(factory, name: str) -> None:
    response = factory().generate(GenerateRequest(prompt="contract probe", max_tokens=1))

    assert isinstance(response, GenerateResponse)
    assert response.backend == name
    assert isinstance(response.text, str)
    assert isinstance(response.tokens, int)
    assert response.tokens >= 0
    assert isinstance(response.elapsed_ms, int)
    assert response.elapsed_ms >= 0
    assert isinstance(response.metadata, dict)

