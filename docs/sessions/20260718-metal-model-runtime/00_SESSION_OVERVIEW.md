# Metal Model Runtime Session

## Objective

Build a production-grade, research-first Metal execution system for agentic Qwen models,
mixture-of-experts models, discrete diffusion models, and justified exotic architectures.
The system must preserve hexagonal boundaries while using strong native bindings and shared,
ownership-safe tensor layouts.

## Approved Direction

The Rust domain core owns model and operator plans, state contracts, kernel selection, tuning
records, correctness policy, and evaluation. Metal, Zig, Mojo, C, Nim, and Go are specialized
native leaves selected only where measured evidence supports them. Python and TypeScript remain
thin dataset, UX, and integration edges.

## Success Criteria

- One typed execution plan covers dense, MoE, recurrent, diffusion, ternary, and hybrid models.
- Kernel selection is deterministic, observable, cacheable, and correctness-gated.
- Qwen agentic coding is the primary workload; ZAYA, LFM, Bonsai, DeepSeek, Mamba/Jamba/RWKV,
  and diffusion families provide non-Qwen acceptance coverage.
- Native bindings share one versioned ownership-safe ABI.
- Quality, latency, throughput, memory, energy, and stability regressions fail CI.
- Every claimed optimization has a scalar/reference oracle and reproducible benchmark evidence.

## Starting Evidence

- The workspace test run currently fails in eval-harness because public scoring and loader APIs
  are missing or inconsistent.
- Tree-attention has mask, offset, and sibling-isolation risks requiring oracle tests.
- Speculative decoding lacks a complete proposal path and model-owned execution state.
- C and Zig bindings have allocator, partial-allocation, and version-integration risks.
- Existing quality and NIAH evidence is insufficient for regression governance.

## Operating Policy

Research, specify, implement with tests, validate, benchmark, review, and fix forward. No legacy
shims, silent fallbacks, unmeasured backend selection, or unsupported performance claims.
