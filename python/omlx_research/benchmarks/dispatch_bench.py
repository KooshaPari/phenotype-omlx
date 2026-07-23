"""Multi-engine dispatch benchmark — measures dispatch overhead vs single-engine baseline.

Routes requests across mock backends (MLX, vLLM, SGLang, llama.cpp) using
different routing strategies and reports per-request dispatch latency,
throughput, and queue depth distribution.

Usage:
    python -m omlx_research.benchmarks.dispatch_bench
    python python/omlx_research/benchmarks/dispatch_bench.py
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Mock backends ────────────────────────────────────────────────────────────


@dataclass
class MockBackend:
    """Simulates a model-serving engine with configurable latency."""

    name: str
    latency_ms: float = 10.0
    request_count: int = 0
    active: int = 0
    peak_active: int = 0
    total_active_time_ms: float = 0.0

    async def generate(self, prompt: str) -> str:
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        self.request_count += 1
        t0 = time.perf_counter()
        await asyncio.sleep(self.latency_ms / 1000)
        self.total_active_time_ms += (time.perf_counter() - t0) * 1000
        self.active -= 1
        return f"response from {self.name}"

    def reset(self) -> None:
        self.request_count = 0
        self.active = 0
        self.peak_active = 0
        self.total_active_time_ms = 0.0


# ── Metrics ──────────────────────────────────────────────────────────────────


@dataclass
class DispatchMetrics:
    strategy: str = ""
    total_requests: int = 0
    total_dispatch_us: float = 0.0
    dispatch_latencies_us: list[float] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    queue_depths: list[int] = field(default_factory=list)
    backend_assignments: dict[str, int] = field(default_factory=dict)

    @property
    def avg_dispatch_us(self) -> float:
        return self.total_dispatch_us / max(self.total_requests, 1)

    @property
    def p50_dispatch_us(self) -> float:
        return (
            statistics.median(self.dispatch_latencies_us)
            if self.dispatch_latencies_us
            else 0.0
        )

    @property
    def p99_dispatch_us(self) -> float:
        if not self.dispatch_latencies_us:
            return 0.0
        s = sorted(self.dispatch_latencies_us)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]

    @property
    def throughput_rps(self) -> float:
        if self.total_elapsed_ms <= 0:
            return 0.0
        return self.total_requests / (self.total_elapsed_ms / 1000)

    @property
    def avg_queue_depth(self) -> float:
        return statistics.mean(self.queue_depths) if self.queue_depths else 0.0

    @property
    def max_queue_depth(self) -> int:
        return max(self.queue_depths) if self.queue_depths else 0


# ── Routing strategies ──────────────────────────────────────────────────────


class RoutingStrategy(str, Enum):
    SINGLE = "single_engine"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    LEAST_LATENCY = "least_latency"


def _measure_dispatch_ns() -> int:
    """Return current time in nanoseconds for high-precision measurement."""
    return time.perf_counter_ns()


async def dispatch_single(
    backend: MockBackend,
    prompt: str,
    queue_depths: list[int],
) -> tuple[float, str]:
    """Dispatch all requests to a single engine (baseline)."""
    start = _measure_dispatch_ns()
    queue_depths.append(backend.active)
    dispatch_us = (_measure_dispatch_ns() - start) / 1000
    result = await backend.generate(prompt)
    return dispatch_us, result


async def dispatch_round_robin(
    backends: list[MockBackend],
    idx: int,
    prompt: str,
    queue_depths: list[int],
) -> tuple[float, str]:
    """Route to engines in fixed round-robin order."""
    start = _measure_dispatch_ns()
    backend = backends[idx % len(backends)]
    queue_depths.append(backend.active)
    dispatch_us = (_measure_dispatch_ns() - start) / 1000
    result = await backend.generate(prompt)
    return dispatch_us, result


async def dispatch_least_loaded(
    backends: list[MockBackend],
    prompt: str,
    queue_depths: list[int],
) -> tuple[float, str]:
    """Route to the engine with fewest active requests."""
    start = _measure_dispatch_ns()
    # Find least loaded (ties broken by first occurrence)
    best = min(backends, key=lambda b: b.active)
    total_active = sum(b.active for b in backends)
    queue_depths.append(total_active)
    dispatch_us = (_measure_dispatch_ns() - start) / 1000
    result = await best.generate(prompt)
    return dispatch_us, result


# ── Benchmark runners ───────────────────────────────────────────────────────


async def _run_benchmark(
    strategy: RoutingStrategy,
    backends: list[MockBackend],
    n: int,
    concurrency: int = 1,
) -> DispatchMetrics:
    """Run a benchmark with the given strategy and return metrics."""
    for b in backends:
        b.reset()

    metrics = DispatchMetrics(strategy=strategy.value)
    semaphore = asyncio.Semaphore(concurrency)
    queue_depths: list[int] = []
    lock = asyncio.Lock()

    async def _task(idx: int) -> None:
        prompt = f"benchmark request {idx}"
        async with semaphore:
            if strategy == RoutingStrategy.SINGLE:
                d_us, _ = await dispatch_single(backends[0], prompt, queue_depths)
            elif strategy == RoutingStrategy.ROUND_ROBIN:
                d_us, _ = await dispatch_round_robin(
                    backends, idx, prompt, queue_depths
                )
            elif strategy == RoutingStrategy.LEAST_LOADED:
                d_us, _ = await dispatch_least_loaded(backends, prompt, queue_depths)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            async with lock:
                metrics.total_dispatch_us += d_us
                metrics.dispatch_latencies_us.append(d_us)
                metrics.total_requests += 1

    t0 = time.perf_counter()
    await asyncio.gather(*[_task(i) for i in range(n)])
    metrics.total_elapsed_ms = (time.perf_counter() - t0) * 1000
    metrics.queue_depths = queue_depths

    for b in backends:
        metrics.backend_assignments[b.name] = b.request_count

    return metrics


# ── Reporting ────────────────────────────────────────────────────────────────


def _format_report(results: list[DispatchMetrics]) -> str:
    """Format a comparison report."""
    lines = [
        "=" * 72,
        "  Multi-Engine Dispatch Benchmark",
        "=" * 72,
        "",
    ]

    # Summary table header
    lines.append(
        f"{'Strategy':<20} {'μs/req':>10} {'p50 μs':>10} {'p99 μs':>10} "
        f"{'rps':>10} {'maxQ':>6} {'reqs':>6}"
    )
    lines.append("-" * 72)

    for m in results:
        lines.append(
            f"{m.strategy:<20} {m.avg_dispatch_us:>10.1f} {m.p50_dispatch_us:>10.1f} "
            f"{m.p99_dispatch_us:>10.1f} {m.throughput_rps:>10.0f} "
            f"{m.max_queue_depth:>6} {m.total_requests:>6}"
        )

    lines.append("")

    # Per-backend assignment breakdown
    lines.append("Backend Assignment Breakdown:")
    lines.append("-" * 40)
    baseline = results[0] if results else None
    for m in results:
        lines.append(f"  [{m.strategy}]")
        for bname, count in m.backend_assignments.items():
            pct = (count / max(m.total_requests, 1)) * 100
            lines.append(f"    {bname}: {count:>5} ({pct:>5.1f}%)")
    lines.append("")

    # Overhead analysis
    if baseline and len(results) > 1:
        single_us = baseline.avg_dispatch_us
        lines.append("Dispatch Overhead Analysis (vs single-engine baseline):")
        lines.append("-" * 50)
        for m in results[1:]:
            overhead = m.avg_dispatch_us - single_us
            overhead_pct = (overhead / max(single_us, 0.01)) * 100
            lines.append(
                f"  {m.strategy}: +{overhead:.1f}μs/req (+{overhead_pct:.1f}%)"
            )
        lines.append("")

    # Queue depth distribution
    lines.append("Queue Depth Distribution:")
    lines.append("-" * 40)
    for m in results:
        if m.queue_depths:
            lines.append(f"  [{m.strategy}]")
            lines.append(f"    avg={m.avg_queue_depth:.1f}  max={m.max_queue_depth}")
            # Histogram buckets
            buckets = {0: 0, 1: 0, 2: 0, 3: 0, "4+": 0}
            for d in m.queue_depths:
                if d <= 3:
                    buckets[d] += 1
                else:
                    buckets["4+"] += 1
            total = len(m.queue_depths)
            hist = " ".join(f"{k}:{v * 100 // total}%" for k, v in buckets.items())
            lines.append(f"    distribution: {hist}")
    lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    engines = [
        MockBackend("mlx", latency_ms=8.0),
        MockBackend("vllm", latency_ms=12.0),
        MockBackend("sglang", latency_ms=10.0),
        MockBackend("llamacpp", latency_ms=15.0),
    ]

    n_requests = 200
    concurrency = 8

    print(f"Engines: {[e.name for e in engines]}")
    print(f"Requests: {n_requests}, Concurrency: {concurrency}")
    print(f"Engine latencies: {[f'{e.name}={e.latency_ms}ms' for e in engines]}")
    print()

    results: list[DispatchMetrics] = []

    # 1. Single-engine baseline
    m_single = await _run_benchmark(
        RoutingStrategy.SINGLE, engines, n_requests, concurrency
    )
    results.append(m_single)

    # 2. Round-robin
    m_rr = await _run_benchmark(
        RoutingStrategy.ROUND_ROBIN, engines, n_requests, concurrency
    )
    results.append(m_rr)

    # 3. Least-loaded
    m_ll = await _run_benchmark(
        RoutingStrategy.LEAST_LOADED, engines, n_requests, concurrency
    )
    results.append(m_ll)

    report = _format_report(results)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
