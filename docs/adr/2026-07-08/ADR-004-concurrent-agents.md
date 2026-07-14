# ADR-004: Concurrent multi-agent execution (LatentMAS / TiDAR / SSD / JetSpec)

**Date:** 2026-07-08
**Status:** Accepted
**Related:** ADR-002, `python/omlx_research/agents/`

## Context

Four research agents, each with its own internal model:

- **LatentMAS** — multi-agent latent reasoning (Gen-Verse/LatentMAS)
- **TiDAR** — hybrid AR + diffusion generation (irfannaqieb/TiDAR)
- **SSD** — self-speculative decoding (tanishqkumar/ssd, CUDA reference)
- **JetSpec** — draft-head tree speculative decoding (hao-ai-lab/JetSpec)

Each agent is normally a single-threaded loop. The user asked for concurrent
execution so that, e.g., a LatentMAS fan-out can run alongside a TiDAR
diffusion refinement without blocking.

## Decision

Adopt a `ConcurrentScheduler` (`python/omlx_research/agents/scheduler.py`)
that:

1. Builds a DAG of agent invocations (`Strategy::Sequential | FanOut | Reduce | DAG`).
2. Runs the DAG with `asyncio.gather` for I/O-bound work and a Rust
   thread pool (`perf-core/concurrent-exec/`) for CPU-bound work.
3. Each agent exposes `async def step(prompt, state) -> result`, so the
   scheduler can compose any mix of agents without knowing their internals.
4. State is passed between agents as a plain dict (no protocol changes
   needed when an agent is swapped).

The Rust `concurrent-exec` crate provides:

- A plan builder (`plan.rs`) that translates the Python DAG into Rust
  tasks pinned to performance cores where possible.
- Adapter modules per agent (`latentmas.rs`, `tidar.rs`, `ssd.rs`,
  `jetspec.rs`) that own the per-agent state.

## Consequences

**Positive**

- The four agents can be composed freely (LatentMAS → TiDAR, or
  SSD ∥ JetSpec, etc.) without rewriting either.
- The Rust plan builder keeps the CPU-bound work off the GIL.
- The Python surface stays idiomatic (`asyncio.gather`).

**Negative**

- `asyncio.gather` is I/O-bound by default. CPU-bound work inside an
  agent (e.g., SSD's draft loop) still blocks the event loop unless
  the agent is in the Rust tier.
- Cross-agent state passing via dict is loosely typed.

**Mitigations**

- The Rust adapter modules absorb the CPU-bound hot path. The Python
  side is a thin async wrapper.
- The dict state is documented in `python/omlx_research/agents/scheduler.py`.
