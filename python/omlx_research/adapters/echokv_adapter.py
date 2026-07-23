"""EchoKV adapter — wraps the Rust EchoKV pyo3 bindings behind CachePort."""

from __future__ import annotations

import logging
from typing import Any

from ..ports.inference import CachePort

logger = logging.getLogger(__name__)


class EchoKVAdapter(CachePort):
    """Adapter that bridges the Rust EchoKV cache to the CachePort contract.

    Usage::

        cache = EchoKVAdapter(max_size=512)
        cache.insert(token_id=42, attention_weight=0.95)
        evicted = cache.evict()
    """

    def __init__(self, max_size: int = 512) -> None:
        self._max_size = max_size
        self._store: dict[int, float] = {}
        self._kv_engine: Any = None

    def _ensure_engine(self) -> Any:
        """Lazy-load the Rust EchoKV engine via pyo3 bindings."""
        if self._kv_engine is not None:
            return self._kv_engine
        try:
            from omlx_perf_core import echo_kv  # noqa: F401

            self._kv_engine = echo_kv
            return self._kv_engine
        except ImportError:
            logger.debug(
                "EchoKV pyo3 extension not available; using in-memory fallback"
            )
            return None

    def insert(self, token_id: int, attention_weight: float) -> None:
        engine = self._ensure_engine()
        if engine is not None:
            engine.insert(token_id, attention_weight)
        else:
            self._store[token_id] = attention_weight
            if len(self._store) > self._max_size:
                self._evict_python()

    def evict(self) -> list:
        engine = self._ensure_engine()
        if engine is not None:
            return engine.evict()
        return self._evict_python()

    def _evict_python(self) -> list:
        """Fallback eviction: drop lowest-attention entries until within budget."""
        if not self._store:
            return []
        sorted_keys = sorted(self._store, key=lambda k: self._store[k])
        evicted: list = []
        while len(self._store) > self._max_size and sorted_keys:
            key = sorted_keys.pop(0)
            evicted.append(key)
            del self._store[key]
        return evicted
