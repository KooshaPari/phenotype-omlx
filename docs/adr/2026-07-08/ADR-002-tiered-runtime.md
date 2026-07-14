# ADR-002: phenotype-omlx tiered runtime (Rust perf-core + Python surface + OMLX framework)

**Date:** 2026-07-08
**Status:** Accepted
**Supersedes:** none
**Related:** ADR-001, `docs/adr/2026-06-18/ADR-035A-hwledger-reclassification.md`

## Context

The upstream OMLX fork we own is a Python-only Electron app that uses the
bundled MLX framework on Apple Silicon. It works well for single-stream
inference, but three problems emerge when we try to add research tooling
(spec-decode, multi-agent, multi-engine):

1. **Hot-path latency.** Speculative decoding verification, TurboQuant
   pack/unpack, and tree-attention all run per-token. Pure Python adds
   ~50-200 µs per token — small per call, but cumulative at 50+ tok/s.
2. **Cross-platform support.** Upstream is Apple-Silicon-only. To support
   Windows and Linux clients, we need a runtime that compiles to all three
   platforms without rewriting the OMLX framework.
3. **Multi-engine fan-out.** The research agents (LatentMAS, TiDAR, SSD,
   JetSpec) want to talk to MLX, vLLM, TensorRT, SGLang, and llama.cpp
   interchangeably. The current OMLX architecture is hard-bound to MLX.

## Decision

We adopt a three-tier runtime:

1. **OMLX framework tier** (Python 3.11 + MLX, in `/Applications/oMLX.app`)
   — untouched. We inject one file (`mlx/nn/layers/turbo_kv_cache.py`) into
   the bundled site-packages to enable TurboQuant+ KV cache compression.
2. **Python surface tier** (`python/omlx_research/`) — uniform Python API
   for backends, engines, agents, CLI, and the local web admin. Optional
   pyo3 FFI to the Rust perf-core.
3. **Rust perf-core tier** (`perf-core/`, 5-crate workspace) — hot-path
   implementations of spec-decode, concurrent-exec, turbo-quant,
   tree-attention, and fleet-proto. Compiles to `.so` / `.dylib` / `.dll`.

The three tiers communicate via:
- **Framework ↔ Python:** `PYTHONPATH` ordering. The OMLX framework's
  site-packages is first; the research packages come after.
- **Python ↔ Rust:** pyo3 FFI module `_phenotype_omlx_core`. Optional —
  Python falls back to pure-Python implementations if the .so isn't built.

## Consequences

**Positive**

- Hot-path latency drops 2-5× on the Rust side (measured in
  `perf-core/turbo-quant` unit tests).
- Cross-platform support comes for free — the same Rust workspace
  compiles to a Windows .dll, a Linux .so, and a macOS .dylib.
- The OMLX app bundle is untouched except for one injected file
  (re-copyable on every OMLX update).
- Existing OMLX users see no behavior change unless they opt in.

**Negative**

- Build complexity goes up: now need a Rust toolchain alongside the
  Python venv.
- pyo3 is sensitive to Python minor versions — the FFI is built per
  Python (3.11 for OMLX, 3.12 for the system venv).
- Two source languages to maintain.

**Mitigations**

- `scripts/phenotype-omlx-ready` is idempotent and verifies the build on
  every invocation.
- The Rust crates are small and self-contained (no unsafe except in the
  pack/unpack kernels).

## Alternatives considered

| Alternative | Rejected because |
| --- | --- |
| Pure Python (Cython) | Cython's Metal story is poor; Cython-on-Apple-Silicon requires Rosetta. |
| Pure C++ | Loses the Python surface; doubles the maintenance burden. |
| Pure Mojo | Mojo is pre-1.0 and the interop story with MLX is unproven. |
| Pure Zig | No ecosystem for HTTP / JSON / model serving. |
| Pure Go | No MLX / Metal story; the per-token hot path is still GIL-bound. |
