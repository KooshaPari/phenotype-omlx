"""
Hybrid Backend Demonstration — runs ALL strategies concurrently.

Domain decomposition (the answer to "use all backends + all strategies together"):
  - BACKENDS  = model-serving runtimes (mlx-metal, sglang, vllm, tensorrt, llamacpp)
  - STRATEGIES = multi-agent/decoding wrappers (latentmas, tidar, jetspec, ssd)
  - PIPELINE  = sequence of (phase, backend) pairs each strategy declares

Run ALL strategies concurrently — each picks its native backend per phase.
No pytorch fallback, no llamacpp-as-strategy — only the strategy's declared
compatible_backends are used.
"""
import sys, time, asyncio
sys.path.insert(0, "/Users/kooshapari/CodeProjects/Phenotype/repos/phenotype-omlx/python")

from omlx_research.nanovm import (
    discover_plugins, list_available_backends, list_available_strategies,
    get_registry,
)
from omlx_research.hardware import detect_hardware
from omlx_research.hybrid.orchestrator import get_executor


# ── 1. Plugin discovery ─────────────────────────────────────────────────
print("=" * 64)
print("STEP 1 — NanoVM plugin discovery (BACKEND + STRATEGY kinds)")
print("=" * 64)
discover_plugins()

backends = list_available_backends(include_unloadable=True)
print(f"\n  Backends ({len(backends)} total):")
for b in backends:
    status = "loadable" if b["loadable"] else f"unloadable: {b['reason'][:40]}"
    print(f"    - {b['name']:14s} runtime={b['runtime']:10s} prio={b['priority']:3d} ({status})")

strat_names = list_available_strategies()
print(f"\n  Strategies ({len(strat_names)}): {strat_names}")


# ── 2. Hardware detection ───────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 2 — Hardware detection (cross-platform)")
print("=" * 64)
hw = detect_hardware()
print(f"  CPU cores:    {hw.cpu_count_logical} logical / {hw.cpu_count_physical} physical")
print(f"  Memory:       {hw.memory_total_gb:.1f} GB")
print(f"  Accelerators: {[a.name for a in hw.accelerators]}")


# ── 3. Single strategy: latentmas ───────────────────────────────────────
print()
print("=" * 64)
print("STEP 3 — Single strategy: latentmas (debate→propose→vote)")
print("=" * 64)

executor = get_executor()
registry = get_registry()
spec = registry.get("latentmas")
if spec:
    print(f"\n  latentmas spec (from registry):")
    print(f"    phases:              {spec.phases}")
    print(f"    compatible_backends: {spec.compatible_backends}")
    print(f"    default_pool:        {spec.default_pool}")
    print(f"    parallel:            {spec.parallel}")

prompt = "What is the capital of France?"
result = asyncio.run(executor.run("latentmas", prompt, max_tokens=20, temperature=0.0))
print(f"\n  Result:")
print(f"    strategy:        {result.strategy}")
print(f"    text:            {result.text[:80]!r}")
print(f"    elapsed_ms:      {result.elapsed_ms:.1f}")
print(f"    parallel:        {result.parallel}")
print(f"    backends_used:   {result.backends_used}")
print(f"    fallback_used:   {result.fallback_used}")
if result.phase_results:
    print(f"    phase_results:")
    for phase, r in result.phase_results.items():
        print(f"      - {phase:20s} backend={r.backend_name:12s} ok={r.ok} elapsed={r.elapsed_ms:.1f}ms text={r.text[:30]!r}")


# ── 4. ALL strategies concurrently ───────────────────────────────────────
print()
print("=" * 64)
print("STEP 4 — ALL strategies concurrently (latentmas + tidar + jetspec + ssd)")
print("=" * 64)

print(f"\n  Running all {len(strat_names)} strategies in parallel against: {prompt!r}")
t0 = time.perf_counter()
results = asyncio.run(executor.run_all(strat_names, prompt, max_tokens=20, temperature=0.0))
total_elapsed = (time.perf_counter() - t0) * 1000

print(f"\n  All {len(results)} strategies completed in {total_elapsed:.1f}ms (wall clock)")
print()
print(f"  {'Strategy':15s} {'Backend(s)':40s} {'Elapsed':>10s}")
print(f"  {'-'*15} {'-'*40} {'-'*10:>10}")
for r in results:
    backends_str = ", ".join(set(r.backends_used.values())) if r.backends_used else "-"
    print(f"  {r.strategy:15s} {backends_str:40s} {r.elapsed_ms:>9.1f}ms")


# ── 5. Summary ───────────────────────────────────────────────────────────
print()
print("=" * 64)
print("SUMMARY — Strategies × Backends decomposition")
print("=" * 64)
print()
print("  PHASE 1 — Backends (model-serving runtimes):")
for b in backends:
    if b["loadable"]:
        print(f"    {b['name']:14s} runtime={b['runtime']:10s} [loadable]")

print()
print("  PHASE 2 — Strategies (multi-agent/decoding wrappers):")
for sname in strat_names:
    s = registry.get(sname)
    if s:
        print(f"    {s.name:14s} phases={s.phases}")
        print(f"                  default_pool={s.default_pool}")
        print(f"                  compatible={s.compatible_backends}")
        print(f"                  parallel={s.parallel}")

print()
print("  PHASE 3 — Pipeline (strategy phases × backend picks):")
print("    Each strategy declares its phases; orchestrator picks the best")
print("    available backend per phase. parallel=true → all phases run")
print("    concurrently via asyncio.gather.")
print()
print("  RESULT: All strategies ran concurrently above (each picking the")
print("          best native backend available). PyTorch/llama.cpp are NOT")
print("          fallbacks for strategy execution — only the strategy's")
print("          declared compatible_backends are considered.")
