"""Memory profiling for agent execution strategies."""

import asyncio
import tracemalloc
import time
import json


class MockModel:
    def __init__(self, latency_ms=10):
        self.latency_ms = latency_ms

    async def run(self, prompt):
        await asyncio.sleep(self.latency_ms / 1000)
        return f"resp_{len(prompt)}"


def profile_sequential(model, n):
    tracemalloc.start()
    start = time.perf_counter()
    for i in range(n):
        asyncio.run(model.run(f"task_{i}"))
    elapsed = (time.perf_counter() - start) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "strategy": "sequential",
        "tasks": n,
        "elapsed_ms": round(elapsed, 1),
        "current_mb": round(current / 1024 / 1024, 2),
        "peak_mb": round(peak / 1024 / 1024, 2),
    }


def profile_parallel(model, n):
    tracemalloc.start()

    async def run_all():
        return await asyncio.gather(*[model.run(f"task_{i}") for i in range(n)])

    start = time.perf_counter()
    asyncio.run(run_all())
    elapsed = (time.perf_counter() - start) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "strategy": "parallel",
        "tasks": n,
        "elapsed_ms": round(elapsed, 1),
        "current_mb": round(current / 1024 / 1024, 2),
        "peak_mb": round(peak / 1024 / 1024, 2),
    }


if __name__ == "__main__":
    results = []
    for n in [10, 50, 100]:
        m = MockModel(5)
        results.append(profile_sequential(m, n))
        results.append(profile_parallel(m, n))

    print(f"{'Strategy':<12} {'Tasks':>6} {'Time':>10} {'Current':>10} {'Peak':>10}")
    print("-" * 52)
    for r in results:
        print(
            f"{r['strategy']:<12} {r['tasks']:>6} {r['elapsed_ms']:>8.1f}ms "
            f"{r['current_mb']:>8.2f}MB {r['peak_mb']:>8.2f}MB"
        )

    with open("python/omlx_research/benchmarks/memory_profile_results.json", "w") as f:
        json.dump(results, f, indent=2)
