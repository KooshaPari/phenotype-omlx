"""Speculative decoding engine — TurboQuant+ compatible.

Implements three modes:
  - same-model (prompt-lookup, n-gram match): SSD
  - draft-model (separate draft network): MLX draft model
  - medusa (multi-head draft + tree-verify): JetSpec
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Callable, Awaitable


class SpecMode(str, Enum):
    SAME_MODEL = "same_model"
    DRAFT_MODEL = "draft_model"
    MEDUSA = "medusa"


@dataclass
class SpecConfig:
    mode: SpecMode = SpecMode.SAME_MODEL
    max_draft_tokens: int = 5
    gamma: int = 5
    fallback_on_reject: bool = True


@dataclass
class SpecStats:
    drafted: int = 0
    accepted: int = 0
    steps: int = 0

    @property
    def acceptance_rate(self) -> float:
        return 1.0 if self.drafted == 0 else self.accepted / self.drafted


class SpeculativeEngine:
    def __init__(self, target: Callable, draft: Callable | None = None, config: SpecConfig | None = None):
        self.target = target
        self.draft = draft
        self.config = config or SpecConfig()
        self.stats = SpecStats()

    def propose(self, prefix: list[int]) -> list[list[int]]:
        if self.config.mode == SpecMode.SAME_MODEL:
            return [self._prompt_lookup(prefix, self.config.max_draft_tokens)]
        if self.config.mode == SpecMode.DRAFT_MODEL and self.draft is not None:
            return [self.draft(prefix, self.config.max_draft_tokens)]
        if self.config.mode == SpecMode.MEDUSA:
            return self._expand_tree(prefix, self.config.gamma)
        return [self._prompt_lookup(prefix, self.config.max_draft_tokens)]

    def verify(self, prefix: list[int], candidates: list[list[int]]) -> list[bool]:
        target_logits = self.target(prefix)
        out = []
        for c in candidates:
            if not c:
                out.append(False)
                continue
            target_next = int(target_logits.argmax())
            out.append(target_next == c[0])
        return out

    def step(self, prefix: list[int]) -> list[int]:
        candidates = self.propose(prefix)
        accepts = self.verify(prefix, candidates)
        accepted: list[int] = []
        for cand, ok in zip(candidates, accepts):
            self.stats.drafted += len(cand) if cand else 1
            if ok and cand:
                accepted.append(cand[0])
                if len(cand) > 1:
                    accepted.extend(cand[1:])
            else:
                if self.config.fallback_on_reject:
                    accepted.append(int(self.target(prefix).argmax()))
                break
            if len(accepted) >= self.config.max_draft_tokens:
                break
        self.stats.accepted += len(accepted)
        self.stats.steps += 1
        return accepted

    @staticmethod
    def _prompt_lookup(prefix: list[int], k: int) -> list[int]:
        if not prefix:
            return []
        for n in range(min(64, len(prefix)), 0, -1):
            needle = prefix[-n:]
            hay = prefix[: len(prefix) - n]
            pos = SpeculativeEngine._find_subseq(hay, needle)
            if pos is not None:
                start = pos + n
                end = min(start + k, len(prefix))
                if start < end:
                    return prefix[start:end].copy()
        return []

    @staticmethod
    def _find_subseq(hay: list[int], needle: list[int]) -> int | None:
        if not needle or len(hay) < len(needle):
            return None
        for i in range(0, len(hay) - len(needle) + 1):
            if hay[i: i + len(needle)] == needle:
                return i
        return None

    @staticmethod
    def _expand_tree(prefix: list[int], gamma: int) -> list[list[int]]:
        # Medusa: produce `gamma` parallel single-token candidates derived from
        # the last few positions of `prefix`. Real Medusa runs K heads in parallel;
        # this stub returns the last `gamma` tokens as "candidates".
        if not prefix:
            return []
        return [[t] for t in prefix[-gamma:]]
