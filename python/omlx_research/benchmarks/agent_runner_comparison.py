"""Compare all concurrent agent runner strategies with mock backends.

Exercises the real runner APIs from omlx_research.agents:
  - LatentMasRunner  — N latent agents fanned out on one prompt
  - SsdRunner        — self-speculative prompt-lookup draft/verify
  - TidarRunner      — diffusion-draft + AR-verify hybrid loop
  - JetSpecRunner    — tree-attention speculative decoding

Each benchmark compares the runner's native concurrency model against a
sequential baseline to quantify the speedup.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Awaitable, Callable

from omlx_research.agents.jetspec_runner import JetSpecRunner, jetspec_draft_tree
from omlx_research.agents.latentmas_runner import LatentMasRunner, latentmas_fanout
from omlx_research.agents.ssd_runner import SsdRunner
from omlx_research.agents.scheduler import ConcurrentScheduler, Strategy
from omlx_research.agents.tidar_runner import TidarRunner

VOCAB_SIZE = 256

# ---------------------------------------------------------------------------
# Mock backends
# ---------------------------------------------------------------------------


class MockAgent:
    """Callable agent for LatentMAS: (prompt, state) -> str."""

    def __init__(self, latency_ms: float = 10.0, name: str = "agent"):
        self.latency_ms = latency_ms
        self.name = name

    async def __call__(self, prompt: str, state: dict) -> str:
        await asyncio.sleep(self.latency_ms / 1000)
        return f"{self.name}:{len(prompt)}"


class MockTarget:
    """Logit generator for SsdRunner / JetSpecRunner: prefix -> list[float]."""

    def __init__(self, latency_ms: float = 10.0):
        self.latency_ms = latency_ms
        self.calls = 0

    async def __call__(self, prefix: list[int]) -> list[float]:
        await asyncio.sleep(self.latency_ms / 1000)
        self.calls += 1
        logits = [0.0] * VOCAB_SIZE
        if prefix:
            logits[prefix[-1] % VOCAB_SIZE] = 1.0
        else:
            logits[0] = 1.0
        return logits


class MockTidarComponents:
    """base_lm / drafter / verifier for TidarRunner."""

    def __init__(self, latency_ms: float = 10.0):
        self.latency_ms = latency_ms
        self.draft_len = 4

    async def base_lm(self, prefix: list[int]) -> list[float]:
        await asyncio.sleep(self.latency_ms / 1000)
        logits = [0.0] * VOCAB_SIZE
        if prefix:
            logits[prefix[-1] % VOCAB_SIZE] = 1.0
        else:
            logits[0] = 1.0
        return logits

    async def drafter(self, prefix: list[int]) -> list[int]:
        await asyncio.sleep(self.latency_ms / 1000)
        base = prefix[-1] if prefix else 0
        return [(base + i + 1) % VOCAB_SIZE for i in range(self.draft_len)]

    async def verifier(self, prefix: list[int], draft: list[int]) -> list[int]:
        await asyncio.sleep(self.latency_ms / 1000)
        return [1] * len(draft)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


def _now_ms() -> float:
    return time.perf_counter() * 1000


async def bench_latentmas(n_tasks: int, latency_ms: float) -> dict:
    agents = [MockAgent(latency_ms, name=f"agent_{i}") for i in range(n_tasks)]

    start = _now_ms()
    for i in range(n_tasks):
        await agents[i](f"task_{i}", {})
    seq_ms = _now_ms() - start

    runner = LatentMasRunner(agents, n_parallel=n_tasks)
    start = _now_ms()
    await runner("shared_prompt", {})
    par_ms = _now_ms() - start

    return {
        "runner": "LatentMAS",
        "n_tasks": n_tasks,
        "sequential_ms": round(seq_ms, 1),
        "parallel_ms": round(par_ms, 1),
        "speedup": round(seq_ms / par_ms, 2) if par_ms > 0 else 0,
    }


async def bench_ssd(n_tokens: int, latency_ms: float) -> dict:
    target = MockTarget(latency_ms)
    runner = SsdRunner(target, gamma=5)

    start = _now_ms()
    prefix: list[int] = []
    for i in range(n_tokens):
        logits = await target(prefix)
        tok = int(max(range(len(logits)), key=lambda i: logits[i]))
        prefix.append(tok)
    seq_ms = _now_ms() - start

    target2 = MockTarget(latency_ms)
    runner2 = SsdRunner(target2, gamma=5)
    start = _now_ms()
    prefix2: list[int] = []
    for _ in range(n_tokens // 5 + 1):
        step_tokens = await runner2.step(prefix2)
        prefix2.extend(step_tokens)
        if len(prefix2) >= n_tokens:
            break
    par_ms = _now_ms() - start

    return {
        "runner": "SSD",
        "n_tasks": n_tokens,
        "sequential_ms": round(seq_ms, 1),
        "parallel_ms": round(par_ms, 1),
        "speedup": round(seq_ms / par_ms, 2) if par_ms > 0 else 0,
    }


async def bench_tidar(n_tokens: int, latency_ms: float) -> dict:
    comps = MockTidarComponents(latency_ms)

    start = _now_ms()
    prefix: list[int] = []
    for i in range(n_tokens):
        logits = await comps.base_lm(prefix)
        tok = int(max(range(len(logits)), key=lambda i: logits[i]))
        prefix.append(tok)
    seq_ms = _now_ms() - start

    runner = TidarRunner(
        base_lm=comps.base_lm,
        drafter=comps.drafter,
        verifier=comps.verifier,
        draft_len=4,
        steps=n_tokens // 4 + 1,
    )
    start = _now_ms()
    await runner([0])
    par_ms = _now_ms() - start

    return {
        "runner": "TiDAR",
        "n_tasks": n_tokens,
        "sequential_ms": round(seq_ms, 1),
        "parallel_ms": round(par_ms, 1),
        "speedup": round(seq_ms / par_ms, 2) if par_ms > 0 else 0,
    }


async def bench_jetspec(n_tokens: int, latency_ms: float) -> dict:
    target = MockTarget(latency_ms)

    start = _now_ms()
    prefix: list[int] = []
    for i in range(n_tokens):
        logits = await target(prefix)
        tok = int(max(range(len(logits)), key=lambda i: logits[i]))
        prefix.append(tok)
    seq_ms = _now_ms() - start

    target2 = MockTarget(latency_ms)
    tree = jetspec_draft_tree(
        width=4, depth=2, head_fn=lambda i: (i + 1) % VOCAB_SIZE, prefix=[0]
    )
    runner = JetSpecRunner(target2, draft_tree=tree, width=4, depth=2)
    start = _now_ms()
    prefix2: list[int] = []
    while len(prefix2) < n_tokens:
        accepted = await runner.step(prefix2)
        prefix2.extend(accepted)
    par_ms = _now_ms() - start

    return {
        "runner": "JetSpec",
        "n_tasks": n_tokens,
        "sequential_ms": round(seq_ms, 1),
        "parallel_ms": round(par_ms, 1),
        "speedup": round(seq_ms / par_ms, 2) if par_ms > 0 else 0,
    }


async def bench_scheduler_strategies(n_agents: int, latency_ms: float) -> dict:
    agents = {
        f"agent_{i}": MockAgent(latency_ms, name=f"agent_{i}") for i in range(n_agents)
    }
    results = {}

    for strat_name, strat in [
        ("FANOUT", Strategy.FANOUT),
        ("CHAIN", Strategy.CHAIN),
        ("ROUNDROBIN", Strategy.ROUNDROBIN),
    ]:
        sched = ConcurrentScheduler(agents, strategy=strat, max_concurrency=n_agents)
        start = _now_ms()
        await sched.dispatch("task", {})
        ms = _now_ms() - start
        results[strat_name] = round(ms, 1)

    results["FALLBACK"] = None
    sched_fb = ConcurrentScheduler(
        agents, strategy=Strategy.FALLBACK, max_concurrency=n_agents
    )
    start = _now_ms()
    await sched_fb.dispatch("task", {})
    results["FALLBACK"] = round(_now_ms() - start, 1)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    latency_ms = 20.0
    task_counts = [5, 10, 25, 50]

    print("=" * 68)
    print("Agent Runner Comparison Benchmark")
    print(f"Mock backend latency: {latency_ms}ms per call")
    print("=" * 68)

    # ---- Per-runner benchmarks ----
    all_results: list[dict] = []

    header = f"{'Runner':<12} {'Tasks':>6} {'Sequential':>12} {'Parallel':>12} {'Speedup':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for n in task_counts:
        r = await bench_latentmas(n, latency_ms)
        all_results.append(r)
        print(
            f"{r['runner']:<12} {r['n_tasks']:>6} {r['sequential_ms']:>10.1f}ms {r['parallel_ms']:>10.1f}ms {r['speedup']:>9.2f}x"
        )

        r = await bench_ssd(n, latency_ms)
        all_results.append(r)
        print(
            f"{r['runner']:<12} {r['n_tasks']:>6} {r['sequential_ms']:>10.1f}ms {r['parallel_ms']:>10.1f}ms {r['speedup']:>9.2f}x"
        )

        r = await bench_tidar(n, latency_ms)
        all_results.append(r)
        print(
            f"{r['runner']:<12} {r['n_tasks']:>6} {r['sequential_ms']:>10.1f}ms {r['parallel_ms']:>10.1f}ms {r['speedup']:>9.2f}x"
        )

        r = await bench_jetspec(n, latency_ms)
        all_results.append(r)
        print(
            f"{r['runner']:<12} {r['n_tasks']:>6} {r['sequential_ms']:>10.1f}ms {r['parallel_ms']:>10.1f}ms {r['speedup']:>9.2f}x"
        )

        print()

    # ---- Scheduler strategy comparison ----
    print("=" * 68)
    print("ConcurrentScheduler Strategy Comparison (10 agents, 20ms latency)")
    print("=" * 68)
    sched_results = await bench_scheduler_strategies(10, latency_ms)
    for strat, ms in sched_results.items():
        print(f"  {strat:<12} {ms:>8.1f}ms")

    all_results.append({"type": "scheduler_strategies", "agents": 10, **sched_results})

    # ---- Save ----
    out_path = "python/omlx_research/benchmarks/agent_runner_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
