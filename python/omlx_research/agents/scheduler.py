"""Concurrent scheduler — fan-out / chain / fallback across heterogeneous agents."""

from __future__ import annotations
import asyncio
from enum import Enum


class Strategy(str, Enum):
    FANOUT = "fanout"
    CHAIN = "chain"
    FALLBACK = "fallback"
    ROUNDROBIN = "roundrobin"


class ConcurrentScheduler:
    """Run multiple agents concurrently under a single, simple, async interface."""

    def __init__(
        self,
        agents: dict[str, callable],
        strategy: Strategy = Strategy.FANOUT,
        max_concurrency: int = 4,
    ):
        self.agents = agents
        self.strategy = strategy
        self._sem = asyncio.Semaphore(max(1, max_concurrency))
        self._rr_idx = 0

    async def dispatch(self, prompt, state=None, top: int = 1) -> list:
        state = state or {}
        if self.strategy == Strategy.FANOUT:

            async def _run(name, fn):
                async with self._sem:
                    return name, await fn(prompt, state)

            pairs = await asyncio.gather(*[_run(n, f) for n, f in self.agents.items()])
            return [r for _, r in sorted(pairs)]
        if self.strategy == Strategy.CHAIN:
            out = []
            current = prompt
            for name, fn in self.agents.items():
                async with self._sem:
                    res = await fn(current, state)
                out.append(res)
                current = (current, res)
            return out
        if self.strategy == Strategy.FALLBACK:
            for _, fn in self.agents.items():
                try:
                    async with self._sem:
                        res = await fn(prompt, state)
                    return [res]
                except Exception:
                    continue
            return []
        # ROUNDROBIN — rotate through agents.
        keys = list(self.agents.keys())
        name = keys[self._rr_idx % len(keys)]
        self._rr_idx += 1
        return [await self.agents[name](prompt, state)]
