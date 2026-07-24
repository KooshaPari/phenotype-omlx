"""Ports: pure ABC contracts for domain operations (hexagonal architecture)."""

from .inference import (
    InferencePort,
    InferenceRequest,
    InferenceResponse,
    CachePort,
    DispatchPort,
)

__all__ = [
    "InferencePort",
    "InferenceRequest",
    "InferenceResponse",
    "CachePort",
    "DispatchPort",
]
