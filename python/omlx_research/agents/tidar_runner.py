"""TiDAR — Think in Diffusion, Talk in AR. Hybrid draft/verify loop."""

from __future__ import annotations
import asyncio
from typing import Callable, Awaitable


async def tidar_ar_diffusion_loop(
    base_lm: Callable[[list[int]], Awaitable[list[float]]],
    drafter: Callable[[list[int]], Awaitable[list[int]]],
    verifier: Callable[[list[int], list[int]], Awaitable[list[int]]],
    prompt: list[int],
    draft_len: int = 4,
    steps: int = 8,
    mask_token_id: int = 0,
) -> list[int]:
    """Run a TiDAR hybrid pass: draft `draft_len` tokens in parallel via a
    diffusion/parallel-forwards path, then verify with a single AR forward.
    """
    out: list[int] = list(prompt)
    for _ in range(steps):
        # Diffusion draft step — emit N candidate tokens in parallel.
        draft = await drafter(out[-draft_len:])
        # AR verify — single forward pass to confirm the candidates.
        accept = await verifier(out, draft)
        for tok, ok in zip(draft, accept):
            if ok and tok != mask_token_id:
                out.append(tok)
        # Always emit at least one verified token per step.
        if len(out) < len(prompt) + 1:
            logits = await base_lm(out)
            out.append(int(max(range(len(logits)), key=lambda i: logits[i])))
    return out


class TidarRunner:
    def __init__(self, base_lm, drafter, verifier, draft_len: int = 4, steps: int = 8, mask_token_id: int = 0):
        self.base_lm = base_lm
        self.drafter = drafter
        self.verifier = verifier
        self.draft_len = draft_len
        self.steps = steps
        self.mask_token_id = mask_token_id

    async def __call__(self, prompt: list[int]) -> list[int]:
        return await tidar_ar_diffusion_loop(
            self.base_lm, self.drafter, self.verifier,
            prompt, self.draft_len, self.steps, self.mask_token_id,
        )
