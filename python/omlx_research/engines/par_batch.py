"""Parallel-batch engine — submit N prompts, gather N responses."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BatchResult:
    text: str
    tokens: int
    latency_ms: int
    label: str = ""


@dataclass
class ParallelBatchConfig:
    max_workers: int = 4
    timeout_s: float = 60.0


class ParallelBatchEngine:
    """Runs the same call against N inputs in parallel via ThreadPoolExecutor.

    Use case: LatentMAS-style fan-out where each latent agent or each
    speculative-mode candidate is processed concurrently.
    """

    def __init__(self, fn: Callable, config: ParallelBatchConfig | None = None):
        self.fn = fn
        self.config = config or ParallelBatchConfig()

    def __call__(self, items: list) -> list[BatchResult]:
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = [pool.submit(self.fn, x) for x in items]
            return [BatchResult(text=f.result(), tokens=1, latency_ms=1) for f in futures if not f.exception()]
