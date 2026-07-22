"""Pipeline executor — runs STRATEGY pipelines using compatible BACKENDS.

Domain decomposition (matches nanovm.PluginSpec schema):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  BACKEND    (kind=backend)   model-serving runtime, native to one GPU │
  │  STRATEGY   (kind=strategy)  multi-agent / decoding wrapper that       │
  │                              declares PHASES + COMPATIBLE_BACKENDS      │
  │  PIPELINE   (this module)    ordered or parallel execution of phases,  │
  │                              each phase picking a backend from the     │
  │                              shared pool that is:                       │
  │                                (a) compatible (in strategy.compatible) │
  │                                (b) registered and loadable              │
  │                                (c) preferred per phase.default_pool     │
  └─────────────────────────────────────────────────────────────────────────┘

Usage:
    from omlx_research.nanovm import discover_plugins, get_registry
    from omlx_research.hybrid.orchestrator import PipelineExecutor

    discover_plugins()
    executor = PipelineExecutor()
    result = await executor.run("latentmas", "What is the capital of France?")

    # Multi-strategy concurrent:
    results = await executor.run_all(
        strategies=["latentmas", "tidar", "jetspec"],
        prompt="...",
    )
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..nanovm import PluginRegistry, PluginSpec, PluginKind, get_registry

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class BackendRequest:
    """A single request sent to a backend for one pipeline phase."""
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.0
    stop: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendResult:
    """Result of one backend invocation."""
    backend_name: str
    backend_kind: str
    text: str = ""
    tokens: int = 0
    elapsed_ms: float = 0.0
    ok: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "kind": self.backend_kind,
            "text": self.text,
            "tokens": self.tokens,
            "elapsed_ms": self.elapsed_ms,
            "ok": self.ok,
            "error": self.error,
        }


@dataclass
class PipelineResult:
    """Result of running one strategy's pipeline."""
    strategy: str
    prompt: str
    phase_results: dict[str, BackendResult] = field(default_factory=dict)
    text: str = ""
    elapsed_ms: float = 0.0
    parallel: bool = False
    backends_used: dict[str, str] = field(default_factory=dict)
    fallback_used: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "text": self.text,
            "elapsed_ms": self.elapsed_ms,
            "parallel": self.parallel,
            "phases": {
                phase: br.to_dict()
                for phase, br in self.phase_results.items()
            },
            "backends_used": self.backends_used,
            "fallback_used": self.fallback_used,
        }


# ── Backend execution surface ──────────────────────────────────────────────

def _stub_backend_execute(
    backend: PluginSpec, phase: str, request: BackendRequest
) -> BackendResult:
    """Default backend executor — returns a stub result for any backend.
    Real backends would be invoked here; we keep the surface pluggable so
    each `module` attribute in plugin.toml can register its own handler.
    """
    t0 = time.perf_counter()
    text = (
        f"[{backend.name}:{phase}] {request.prompt[:40]} "
        f"({request.max_tokens} tok, T={request.temperature})"
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return BackendResult(
        backend_name=backend.name,
        backend_kind=backend.runtime,
        text=text,
        tokens=request.max_tokens,
        elapsed_ms=elapsed,
        ok=True,
        metadata={"phase": phase, "stub": True},
    )


_EXECUTOR_REGISTRY: dict[str, Any] = {}


def register_executor(plugin_name: str, fn: Any) -> None:
    """Register a custom executor function for a backend/strategy plugin."""
    _EXECUTOR_REGISTRY[plugin_name] = fn


def _run_backend(backend: PluginSpec, phase: str, request: BackendRequest) -> BackendResult:
    fn = _EXECUTOR_REGISTRY.get(backend.name)
    if fn is not None:
        try:
            return fn(backend, phase, request)
        except Exception as e:
            return BackendResult(
                backend_name=backend.name,
                backend_kind=backend.runtime,
                ok=False,
                error=f"{type(e).__name__}: {e}",
            )
    return _stub_backend_execute(backend, phase, request)


# ── Pipeline executor ──────────────────────────────────────────────────────

class PipelineExecutor:
    """Run strategy pipelines using compatible backends from the shared pool.

    Picks the best available backend per phase based on:
      1. strategy.compatible_backends (must be in this list)
      2. phase.default_pool[i] preferred (if specified)
      3. backend.priority (higher wins)
      4. backend.can_load() (must succeed)

    If strategy.parallel:  asyncio.gather all phases
    Else:                  sequential, feeding previous phase text as context
    """

    def __init__(self, registry: PluginRegistry | None = None):
        self.registry = registry or get_registry()

    # ── Backend picking ──────────────────────────────────────────────────

    def available_backends(self) -> list[PluginSpec]:
        """Return registered + loadable BACKEND plugins."""
        return [b for b in self.registry.backends() if b.can_load()[0]]

    def pick_backend(
        self,
        strategy: PluginSpec,
        phase_idx: int,
    ) -> tuple[PluginSpec | None, list[str]]:
        """Pick best backend for `strategy.phases[phase_idx]`.

        Returns (backend, fallback_chain_used). The fallback chain is the list
        of backend names we tried before settling on (or failing).
        """
        if strategy.kind != PluginKind.STRATEGY:
            return None, []
        if not strategy.compatible_backends:
            return None, []

        available = self.available_backends()
        available_by_runtime = {b.runtime: b for b in available}
        available_by_name = {b.name: b for b in available}

        fallback_chain: list[str] = []
        tried: set[str] = set()

        candidates: list[str] = []
        if 0 <= phase_idx < len(strategy.default_pool):
            preferred = strategy.default_pool[phase_idx]
            if preferred and preferred not in tried:
                candidates.append(preferred)
                tried.add(preferred)
        for c in strategy.compatible_backends:
            if c not in tried:
                candidates.append(c)
                tried.add(c)

        best: PluginSpec | None = None
        best_score = -1
        for cand in candidates:
            fallback_chain.append(cand)
            b = available_by_runtime.get(cand) or available_by_name.get(cand)
            if b is None:
                continue
            score = b.priority
            if 0 <= phase_idx < len(strategy.default_pool) and strategy.default_pool[phase_idx] == cand:
                score += 100
            if score > best_score:
                best = b
                best_score = score

        return best, fallback_chain

    # ── Single strategy ─────────────────────────────────────────────────

    async def run(
        self,
        strategy_name: str,
        prompt: str,
        *,
        max_tokens: int = 50,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> PipelineResult:
        """Run one strategy's full pipeline."""
        strategy = self.registry.get(strategy_name)
        if strategy is None or strategy.kind != PluginKind.STRATEGY:
            return PipelineResult(
                strategy=strategy_name,
                prompt=prompt,
                text=f"[error: strategy '{strategy_name}' not registered or not kind=strategy]",
                elapsed_ms=0.0,
            )
        ok, reason = strategy.can_load()
        if not ok:
            return PipelineResult(
                strategy=strategy_name,
                prompt=prompt,
                text=f"[error: strategy '{strategy_name}' unloadable: {reason}]",
                elapsed_ms=0.0,
            )

        t_total = time.perf_counter()
        phases = strategy.phases or ["main"]
        phase_results: dict[str, BackendResult] = {}
        backends_used: dict[str, str] = {}
        fallback_chain: list[str] = []

        async def run_phase(idx: int, phase: str) -> tuple[str, BackendResult, list[str]]:
            backend, fchain = self.pick_backend(strategy, idx)
            if backend is None:
                err_result = BackendResult(
                    backend_name="<none>",
                    backend_kind="<unavailable>",
                    ok=False,
                    error=f"no available backend for phase '{phase}' (compatible={strategy.compatible_backends})",
                )
                return phase, err_result, fchain
            req = BackendRequest(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or [],
                metadata={"strategy": strategy_name, "phase_idx": idx},
            )
            result = await asyncio.to_thread(_run_backend, backend, phase, req)
            return phase, result, fchain

        if strategy.parallel:
            phase_outputs = await asyncio.gather(
                *[run_phase(i, p) for i, p in enumerate(phases)],
                return_exceptions=False,
            )
            for phase, result, fchain in phase_outputs:
                phase_results[phase] = result
                if result.ok:
                    backends_used[phase] = result.backend_name
                fallback_chain.extend(fchain)
        else:
            for i, phase in enumerate(phases):
                _, result, fchain = await run_phase(i, phase)
                phase_results[phase] = result
                if result.ok:
                    backends_used[phase] = result.backend_name
                fallback_chain.extend(fchain)

        # Aggregate: take the last successful phase's text
        final_text = ""
        for phase in phases:
            r = phase_results.get(phase)
            if r and r.ok and r.text:
                final_text = r.text

        elapsed = (time.perf_counter() - t_total) * 1000

        return PipelineResult(
            strategy=strategy_name,
            prompt=prompt,
            phase_results=phase_results,
            text=final_text,
            elapsed_ms=elapsed,
            parallel=strategy.parallel,
            backends_used=backends_used,
            fallback_used=fallback_chain,
            metadata={
                "phases": phases,
                "compatible_backends": strategy.compatible_backends,
                "default_pool": strategy.default_pool,
            },
        )

    # ── Multi-strategy concurrent ────────────────────────────────────────

    async def run_all(
        self,
        strategies: list[str],
        prompt: str,
        *,
        max_tokens: int = 50,
        temperature: float = 0.0,
    ) -> list[PipelineResult]:
        """Run multiple strategies concurrently, each picking its own backends.

        This is the answer to "use ALL strategies at once":
          - latentmas  (debate→propose→vote)   uses mlx-metal on each phase
          - tidar      (draft→diffuse→verify)  uses mlx-metal / sglang
          - jetspec    (tree_draft→verify)     uses mlx-metal
          - ssd        (draft→verify)          uses sglang
        All run in parallel against the same prompt, each on its native backend.
        """
        return await asyncio.gather(
            *[self.run(s, prompt, max_tokens=max_tokens, temperature=temperature)
              for s in strategies],
            return_exceptions=False,
        )


# ── Convenience singletons ─────────────────────────────────────────────────

_default_executor: PipelineExecutor | None = None


def get_executor() -> PipelineExecutor:
    global _default_executor
    if _default_executor is None:
        _default_executor = PipelineExecutor()
    return _default_executor


# ── CLI surface (replaces the old FusionPolicy API) ───────────────────────

async def dispatch(
    strategy_name: str,
    prompt: str,
    **kwargs: Any,
) -> PipelineResult:
    """Run a single strategy. Backward-compat with old `dispatch(req, policy)`."""
    return await get_executor().run(strategy_name, prompt, **kwargs)


async def dispatch_all(
    strategies: list[str],
    prompt: str,
    **kwargs: Any,
) -> list[PipelineResult]:
    """Run multiple strategies concurrently."""
    return await get_executor().run_all(strategies, prompt, **kwargs)


__all__ = [
    "BackendRequest",
    "BackendResult",
    "PipelineResult",
    "PipelineExecutor",
    "get_executor",
    "dispatch",
    "dispatch_all",
    "register_executor",
]