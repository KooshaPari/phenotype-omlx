"""SSD — Self-Speculative Decoding. Same-model draft via n-gram prompt-lookup."""

from __future__ import annotations
from collections import deque
from typing import Callable, Awaitable


class SsdRunner:
    """Prompt-lookup + same-model verification."""

    def __init__(self, target: Callable[[list[int]], Awaitable[list[float]]], gamma: int = 5):
        self.target = target
        self.gamma = gamma
        self.seen: deque[int] = deque(maxlen=4096)

    def _lookup(self, prefix: list[int]) -> list[int]:
        if not prefix:
            return []
        seen = list(self.seen)
        for n in range(min(64, len(prefix)), 0, -1):
            needle = prefix[-n:]
            hay = seen[: len(seen) - len(prefix)]
            for i in range(0, len(hay) - n + 1):
                if hay[i: i + n] == needle:
                    start = i + n
                    end = min(start + self.gamma, len(seen))
                    if start < end:
                        return seen[start:end]
        return []

    async def step(self, prefix: list[int]) -> list[int]:
        draft = self._lookup(prefix)
        if not draft:
            logits = await self.target(prefix)
            tok = int(max(range(len(logits)), key=lambda i: logits[i]))
            return [tok]
        accept = []
        for tok in draft:
            logits = await self.target(prefix + accept)
            argmax = int(max(range(len(logits)), key=lambda i: logits[i]))
            if argmax == tok:
                accept.append(tok)
            else:
                accept.append(argmax)
                break
        return accept
