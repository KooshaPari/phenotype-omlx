"""JetSpec — tree-attention speculative decoding runner."""

from __future__ import annotations
import asyncio
from typing import Callable, Awaitable


def jetspec_draft_tree(width: int, depth: int, head_fn: Callable[[int], int], prefix: list[int]) -> list[list[int]]:
    """Generate a tree of draft tokens.

    `head_fn` returns the Medusa-head prediction given a token index. The
    default head_fn returns the last seen token (which keeps structure but
    yields 0% acceptance — useful for testing mask shapes only).
    """
    if width < 1 or depth < 1:
        return []
    roots = [prefix[-1]] if prefix else [0]
    if depth == 1:
        return [[r] for r in roots[:width]]
    res = []
    for r in roots[:width]:
        inner = jetspec_draft_tree(width, depth - 1, head_fn, prefix + [r])
        res.extend([[r] + x for x in inner])
        res.append([r])
    return res


class JetSpecRunner:
    def __init__(self, target: Callable, draft_tree: list[list[int]], width: int = 4, depth: int = 2):
        self.target = target
        self.draft_tree = draft_tree
        self.width = width
        self.depth = depth

    async def step(self, prefix: list[int]) -> list[int]:
        # Forward pass with explicit tree attention mask.
        accepted: list[int] = []
        for branch in self.draft_tree:
            branch_out = []
            for tok in branch:
                logits = await self.target(prefix + accepted + branch_out)
                argmax = int(max(range(len(logits)), key=lambda i: logits[i]))
                if argmax == tok:
                    branch_out.append(tok)
                else:
                    accepted.extend(branch_out)
                    accepted.append(argmax)
                    return accepted
            accepted.extend(branch_out)
        return accepted
