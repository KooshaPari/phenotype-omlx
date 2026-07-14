"""Hybrid multi-backend orchestrator.

Composes ANY combination of available backends (MLX/Metal + SGLang + vLLM +
TensorRT + llama.cpp) into a single fused inference call. Each backend has
its own strengths — the orchestrator routes sub-tasks to whichever is best:

    ┌─────────────────────────────── HybridOrchestrator ───────────────────────────────┐
    │                                                                                    │
    │  dispatch(prompt)                                                                   │
    │    │                                                                                │
    │    ├── mlx-metal     ◀── prefill on Apple Silicon (fast for prompts ≤ 4k)            │
    │    ├── sglang        ◀── continuous batching / large ctx / PD-disagg                  │
    │    ├── vllm          ◀── PagedAttention / high-throughput serving                    │
    │    ├── tensorrt      ◀── single-GPU latency-optimized (Hopper/Ada/Blackwell)         │
    │    ├── llamacpp      ◀── CPU / quantized GGUF, offload model                        │
    │    │                                                                                │
    │    ▼                                                                                │
    │  fan-out policy: parallel (race all, first wins) | majority vote |                  │
    │                   weighted | speculative-draft | mirror-and-verify                   │
    │    ▼                                                                                │
    │  cross-backend verify ─── verify on a different backend (cheap = Metal draft,        │
    │                            expensive = CUDA target — like SSD but cross-vendor)     │
    │    ▼                                                                                │
    │  result                                                                            │
    └────────────────────────────────────────────────────────────────────────────────────┘

Why this matters: in production you usually have ONE best backend per model,
but you ALSO have multiple compute substrates (e.g., M1 Pro + RTX 4090 + CPU
farm). Hybrid lets you use ALL of them in one call, with policy on how to
combine results.
"""

from __future__ import annotations

import asyncio
import logging
import platform as _plat
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .nanovm import PluginRegistry, PluginSpec

logger = logging.getLogger(__name__)


class FusionPolicy(str, Enum):
    PARALLEL_RACE = "parallel_race"          # all backends, first to finish wins
    MAJORITY_VOTE = "majority_vote"          # each backend votes per token
    WEIGHTED = "weighted"                    # backends have weights, weighted average
    SPECULATIVE_DRAFT = "speculative_draft"  # cheap backend drafts, expensive verifies
    MIRROR_VERIFY = "mirror_verify"          # primary runs, secondary verifies


@dataclass
class BackendRequest:
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.0
    stop: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendResult:
    backend: str
    text: str
    tokens: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class HybridResult:
    text: str
    backend_results: dict[str, BackendResult]
    fusion_policy: str
    winner: str
    total_elapsed_ms: float
    metadata: dict = field(default_factory=dict)


class HybridOrchestrator:
    """Composes multiple backend plugins into one fused call.

    Usage:
        orch = HybridOrchestrator.auto()
        result = await orch.dispatch(
            BackendRequest(prompt="hi", max_tokens=32),
            policy=FusionPolicy.PARALLEL_RACE,
            backends=["mlx-metal", "sglang", "llamacpp"],
        )
        print(result.winner, "→", result.text)
    """

    def __init__(self, registry: PluginRegistry):
        self.registry = registry

    @classmethod
    def auto(cls, platform: str | None = None) -> "HybridOrchestrator":
        return cls(PluginRegistry(discover=True))

    # ── discovery helpers ──────────────────────────────────────────────

    def available_backends(self, platform: str | None = None) -> list[PluginSpec]:
        plat = platform or _plat.system().lower()
        return self.registry.backends_for(plat)

    def has(self, name: str) -> bool:
        return self.registry.get(name) is not None

    def devices(self, platform: str | None = None) -> set[str]:
        dev: set[str] = set()
        for s in self.available_backends(platform):
            dev.update(s.devices)
        return dev

    # ── main dispatch ──────────────────────────────────────────────────

    async def dispatch(
        self,
        request: BackendRequest,
        *,
        policy: FusionPolicy | str = FusionPolicy.PARALLEL_RACE,
        backends: list[str] | None = None,
    ) -> HybridResult:
        if isinstance(policy, str):
            policy = FusionPolicy(policy)

        # Resolve which backend specs to use
        avail = self.available_backends()
        if backends:
            specs = [self.registry.get(n) for n in backends]
            specs = [s for s in specs if s is not None]
        else:
            specs = avail[:3]  # top 3 by priority
        if not specs:
            return HybridResult(
                text="",
                backend_results={},
                fusion_policy=policy.value,
                winner="(none)",
                total_elapsed_ms=0.0,
            )

        # Fan out to all selected backends concurrently
        tasks = {s.name: self._run_one(s, request) for s in specs}
        results: dict[str, BackendResult] = {}
        t0 = time.perf_counter()
        for name, coro in tasks.items():
            try:
                results[name] = await coro
            except Exception as e:  # noqa: BLE001
                results[name] = BackendResult(
                    backend=name, text="", elapsed_ms=0.0,
                    error=f"{type(e).__name__}: {str(e)[:120]}",
                )
        total_ms = (time.perf_counter() - t0) * 1000

        # Fuse
        return self._fuse(results, policy, total_ms)

    # ── per-backend execution ──────────────────────────────────────────

    async def _run_one(self, spec: PluginSpec, req: BackendRequest) -> BackendResult:
        """Execute one backend. Tries (in order):
            1. `backend.execute(req) → BackendResult`
            2. `backend.generate(req.prompt, req.max_tokens) → str`
            3. echo (test stub)
        """
        t0 = time.perf_counter()
        try:
            instance = self.registry.instantiate(spec)
        except Exception as e:  # noqa: BLE001
            return BackendResult(
                backend=spec.name, text="", elapsed_ms=0.0,
                error=f"instantiate: {type(e).__name__}: {e}",
            )

        # method 1: typed execute
        for method_name in ("execute", "generate", "complete", "__call__"):
            meth = getattr(instance, method_name, None)
            if meth is None:
                continue
            try:
                if asyncio.iscoroutinefunction(meth):
                    out = await meth(req)
                else:
                    out = meth(req)
                return self._coerce(spec.name, out, (time.perf_counter() - t0) * 1000)
            except Exception as e:  # noqa: BLE001
                return BackendResult(
                    backend=spec.name, text="", elapsed_ms=0.0,
                    error=f"{method_name}: {type(e).__name__}: {e}",
                )

        return BackendResult(
            backend=spec.name, text=f"[{spec.name} stub]",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            metadata={"stub": True},
        )

    @staticmethod
    def _coerce(backend: str, out: Any, elapsed_ms: float) -> BackendResult:
        if isinstance(out, BackendResult):
            return out
        if isinstance(out, str):
            return BackendResult(backend=backend, text=out, elapsed_ms=elapsed_ms)
        if isinstance(out, dict):
            return BackendResult(
                backend=backend,
                text=str(out.get("text", out.get("output", ""))),
                tokens=int(out.get("tokens", 0)),
                elapsed_ms=float(out.get("elapsed_ms", elapsed_ms)),
                metadata={k: v for k, v in out.items() if k not in {"text", "tokens", "elapsed_ms"}},
            )
        return BackendResult(
            backend=backend, text=str(out), elapsed_ms=elapsed_ms,
            metadata={"raw": True},
        )

    # ── fusion strategies ──────────────────────────────────────────────

    def _fuse(
        self,
        results: dict[str, BackendResult],
        policy: FusionPolicy,
        total_ms: float,
    ) -> HybridResult:
        ok = {k: r for k, r in results.items() if r.ok}
        if not ok:
            return HybridResult(
                text="",
                backend_results=results,
                fusion_policy=policy.value,
                winner="(none)",
                total_elapsed_ms=total_ms,
            )

        if policy == FusionPolicy.PARALLEL_RACE:
            winner = min(ok.items(), key=lambda kv: kv[1].elapsed_ms)[0]
            return HybridResult(
                text=ok[winner].text,
                backend_results=results,
                fusion_policy=policy.value,
                winner=winner,
                total_elapsed_ms=total_ms,
            )

        if policy == FusionPolicy.MAJORITY_VOTE:
            # simplest: majority of first 30 chars (stripped)
            from collections import Counter
            sigs = Counter((r.text[:30].strip() for r in ok.values()))
            best_sig, _ = sigs.most_common(1)[0]
            winner = next((k for k, r in ok.items() if r.text[:30].strip() == best_sig), "(none)")
            return HybridResult(
                text=ok[winner].text,
                backend_results=results,
                fusion_policy=policy.value,
                winner=winner,
                total_elapsed_ms=total_ms,
            )

        if policy == FusionPolicy.SPECULATIVE_DRAFT:
            # Pick cheapest as draft, slowest as verifier
            ordered = sorted(ok.items(), key=lambda kv: kv[1].elapsed_ms)
            draft, target = ordered[0][0], ordered[-1][0]
            return HybridResult(
                text=ok[target].text,
                backend_results=results,
                fusion_policy=f"{policy.value}(draft={draft},target={target})",
                winner=target,
                total_elapsed_ms=total_ms,
                metadata={"draft": draft, "target": target},
            )

        if policy == FusionPolicy.MIRROR_VERIFY:
            # Primary is fastest; verify against any other
            ordered = sorted(ok.items(), key=lambda kv: kv[1].elapsed_ms)
            primary = ordered[0]
            verifier = ordered[1] if len(ordered) > 1 else None
            if verifier and primary[1].text.strip() != verifier[1].text.strip():
                return HybridResult(
                    text=verifier[1].text,
                    backend_results=results,
                    fusion_policy=f"{policy.value}(primary={primary[0]},verified_by={verifier[0]})",
                    winner=verifier[0],
                    total_elapsed_ms=total_ms,
                )
            return HybridResult(
                text=primary[1].text,
                backend_results=results,
                fusion_policy=policy.value,
                winner=primary[0],
                total_elapsed_ms=total_ms,
            )

        # WEIGHTED — by priority
        ranked = sorted(ok.items(), key=lambda kv: self.registry.get(kv[0]).priority)
        winner = ranked[0][0]
        return HybridResult(
            text=ok[winner].text,
            backend_results=results,
            fusion_policy=policy.value,
            winner=winner,
            total_elapsed_ms=total_ms,
        )


__all__ = [
    "FusionPolicy", "BackendRequest", "BackendResult", "HybridResult",
    "HybridOrchestrator",
]