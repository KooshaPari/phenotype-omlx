# Implementation Strategy

## Module Boundaries

- model-plan: pure domain types, validation, serialization, reference execution.
- kernel-registry: candidate metadata, capability matching, selection policy, tuning records.
- metal-runtime: compilation, pipeline cache, command scheduling, buffers, device telemetry.
- model-kernels: attention, MoE, recurrent, diffusion, speculative, and quantized packages.
- native-abi: versioned C-compatible descriptors and ownership tests.
- eval-harness: datasets, scoring, trace replay, reports, and regression policy.

Modules should remain below 350 lines where practical and 500 lines absolutely. Public interfaces
must be narrow; model-specific constants belong in validated plans rather than dispatch code.

## Selection Process

The planner emits operators and state requirements. The registry filters candidates by exact
capabilities, validates cached tuning evidence, and either chooses a proven candidate or runs a
bounded tuning session. Every decision emits a structured trace explaining candidates, rejection
reasons, selected measurements, and fallback policy.

## Correctness and Stability

Write failing contract tests before implementation. Keep scalar reference kernels for oracle use,
not production fallback. Test zero lengths, odd dimensions, partial groups, aliasing, allocation
failure, corrupted descriptors, concurrent invocation, and repeated initialization. Fuzz parsers,
ABI descriptors, packed formats, and scheduler state transitions.

## Performance

Optimize from representative traces: minimize dispatches and intermediate buffers; fuse routing,
quantization, and reductions when numerically safe; specialize by shape buckets; reuse pipeline and
buffer caches; overlap CPU planning with GPU execution; measure tails and energy, not means alone.

## AX, DX, and UX

Provide one CLI for plan inspection, candidate explanation, tuning, benchmark replay, evaluation,
and evidence export. Errors must identify model, operator, shape, backend, candidate, and violated
contract. Reports compare stock and frontier models using the same tasks and scoring contracts.

## Security and Governance

Treat model files, kernel source, and benchmark inputs as untrusted. Bound allocations and compile
time, validate all dimensions, hash artifacts, record provenance, and prohibit arbitrary runtime
kernel source in production policy. Promotion requires signed evidence metadata and reviewable
quality/performance deltas.

## DeltaNet two-pass implementation contract (2026-07-23)

The current `deltanet_step_f32` artifact remains the correctness fallback. The optimized
candidate must add two entry points in the same metallib:

1. `deltanet_state_f32`: grid `(n*n,1,1)`, one thread per `(i,j)`, computes
   `next[i,j] = state[i,j] - beta*k[i]*(k^T*state[:,j]) + beta*v[i]*k[j]`.
2. `deltanet_output_f32`: grid `(n,1,1)`, one thread per output `i`, computes
   `out[i] = sum_j q[j] * next[j,i]`.

The Rust wrapper must encode both dispatches in command order on one command buffer,
reuse existing shared buffers, and select the optimized path only for validated shape
buckets. If either pipeline is unavailable or command completion is non-success, it
must return the existing `Metal` error rather than silently mixing partial state.

Acceptance requires scalar-oracle parity, repeated-state continuity, artifact hash capture,
no allocation growth across 100 calls, and p50/p95 improvement over the locked
`448.292/708.000 us` baseline at head dimension 8. The current one-entry
`deltanet_step_f32` symbol remains unchanged until every gate passes.
