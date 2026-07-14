"""LatentMAS concurrent runner — multiple latent agents in parallel."""

from __future__ import annotations
import asyncio
from typing import Callable, Awaitable


async def latentmas_fanout(
    fns: list[Callable[[str, dict], Awaitable[str]]],
    prompt: str,
    state: dict,
) -> list[str]:
    """Run N latent agents concurrently on the same prompt."""
    coros = [fn(prompt, state) for fn in fns]
    return await asyncio.gather(*coros, return_exceptions=False)


class LatentMasRunner:
    """Adapter for LatentMAS's ModelWrapper.run() concurrent fan-out.

    Real LatentMAS dispatches N latent agents (proposer / verifier / refiner /
    critic) in parallel and merges their hidden-state trajectories. This runner
    is the OMLX-side bridge: it accepts a list of pre-built agent callables and
    lets the OMLX scheduler invoke them concurrently.
    """

    def __init__(self, agents: list, n_parallel: int = 4):
        self.agents = agents
        self.n_parallel = n_parallel
        self._sem = asyncio.Semaphore(n_parallel)

    async def __call__(self, prompt: str, state: dict) -> list[str]:
        async def _one(agent) -> str:
            async with self._sem:
                return await agent(prompt, state)

        return await asyncio.gather(*[_one(a) for a in self.agents])
