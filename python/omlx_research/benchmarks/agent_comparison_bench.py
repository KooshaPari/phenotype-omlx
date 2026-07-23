"""LatentMAS vs single-agent comparison benchmark."""

import asyncio
import time
import json
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    strategy: str
    total_time_ms: float
    avg_latency_ms: float
    throughput_rps: float
    num_tasks: int
    mock_latency_ms: float


class MockModelWrapper:
    """Simulates model inference with configurable latency."""

    def __init__(self, latency_ms: float = 50.0):
        self.latency_ms = latency_ms
        self.call_count = 0

    async def run(self, prompt: str) -> str:
        self.call_count += 1
        await asyncio.sleep(self.latency_ms / 1000)
        return f"response_{self.call_count}"


async def single_agent_sequential(model, prompts):
    """Run all prompts sequentially through a single agent."""
    start = time.perf_counter()
    results = []
    for p in prompts:
        r = await model.run(p)
        results.append(r)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, results


async def latentmas_parallel(model, prompts):
    """Run all prompts concurrently via LatentMAS fanout."""
    start = time.perf_counter()
    tasks = [model.run(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, list(results)


async def latentmas_fanout_adapter(fns, prompts):
    """Run using latentmas_fanout pattern (same prompt, N agents)."""
    start = time.perf_counter()
    results = await asyncio.gather(*[fn(p, {}) for fn, p in zip(fns, prompts)])
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, list(results)


async def run_benchmark():
    prompt_counts = [5, 10, 20, 50]
    latency_ms = 50.0
    results = []

    print("=== LatentMAS vs Single-Agent Benchmark ===\n")
    print(
        f"{'Tasks':>6} {'Sequential':>12} {'Parallel':>12} "
        f"{'Speedup':>10} {'Seq RPS':>10} {'Par RPS':>10}"
    )
    print("-" * 72)

    for n in prompt_counts:
        prompts = [f"task_{i}" for i in range(n)]

        model_seq = MockModelWrapper(latency_ms)
        seq_ms, seq_results = await single_agent_sequential(model_seq, prompts)

        model_par = MockModelWrapper(latency_ms)
        par_ms, par_results = await latentmas_parallel(model_par, prompts)

        speedup = seq_ms / par_ms if par_ms > 0 else 0
        seq_rps = n / (seq_ms / 1000)
        par_rps = n / (par_ms / 1000)

        print(
            f"{n:>6} {seq_ms:>10.1f}ms {par_ms:>10.1f}ms "
            f"{speedup:>9.2f}x {seq_rps:>9.1f} {par_rps:>9.1f}"
        )

        results.append(
            {
                "tasks": n,
                "sequential_ms": round(seq_ms, 1),
                "parallel_ms": round(par_ms, 1),
                "speedup": round(speedup, 2),
                "sequential_rps": round(seq_rps, 1),
                "parallel_rps": round(par_rps, 1),
            }
        )

    print()

    out_path = "python/omlx_research/benchmarks/agent_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
