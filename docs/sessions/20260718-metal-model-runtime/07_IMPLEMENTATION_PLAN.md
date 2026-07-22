# Metal Model Runtime Implementation Plan

> Required execution method: use subagent-driven development for bounded packages, with a
> specification review and code-quality review after each package.

**Goal:** Deliver a correctness-gated Metal runtime that selects specialized kernels for Qwen
agentic, sparse MoE, recurrent-hybrid, diffusion, ternary, and related model families while
preserving one typed Rust domain and one ownership-safe native ABI.

**Architecture:** A pure Rust model-plan domain feeds a deterministic kernel registry. Metal and
polyglot native implementations are leaf candidates, not alternate domain models. Evaluation and
promotion consume structured traces and immutable evidence artifacts.

**Toolchains:** Rust nightly where explicitly required, stable Rust conformance, Metal/MLX, Zig,
Mojo, C, Nim, Go, thin Python 3.14 free-threaded and Bun/TS edges.

---

## Task 1: Repair the evaluation baseline

**Files:** perf-core/eval-harness/src/*, perf-core/eval-harness/Cargo.toml

1. Add failing tests for normalization, exact scoring, choice scoring, report aggregation, and
   malformed records.
2. Implement the missing public APIs and deterministic report types.
3. Replace built-in sample records with explicit file or reader loaders carrying provenance.
4. Run cargo test -p eval-harness and cargo test --workspace --all-targets.
5. Review, commit, and snapshot.

## Task 2: Introduce the model-plan domain and reference interpreter

**Files:** new modules under perf-core/model-plan; perf-core/Cargo.toml

1. Test validation for dense, GQA, MLA, CCA, MoE, recurrent, diffusion, speculative, and
   quantized operators.
2. Implement ModelPlan, OperatorPlan, StatePlan, precision policy, and scheduler policy.
3. Add JSON round-trip, unknown-field rejection, dimension overflow, and invalid dependency tests.
4. Implement a slow reference interpreter for small deterministic tensors.
5. Validate size limits, workspace tests, review, commit, and snapshot.

## Task 3: Add the kernel registry and tuning evidence

**Files:** new modules under perf-core/kernel-registry; integration with model-plan

1. Test exact KernelKey identity, capability filtering, deterministic tie-breaks, expiry, and
   hardware/compiler invalidation.
2. Implement candidate metadata, selector policy, TuningRecord, and ExecutionTrace.
3. Add bounded warmup/measurement APIs with variance and confidence metadata.
4. Add evidence serialization and human-readable candidate rejection explanations.
5. Run tests, review, commit, and snapshot.

## Task 4: Correct attention and speculative state

**Files:** perf-core/tree-attention/**, perf-core/spec-decode/**

1. Add scalar oracle tests for mask orientation, sibling isolation, offsets, odd sizes, and trees.
2. Fix tree planning and causal mask generation forward across all callers.
3. Add proposal, verification, acceptance, and per-layer state contracts.
4. Add cancellation, zero-proposal, and malformed-state tests.
5. Benchmark, review, commit, and snapshot.

## Task 5: Establish Native ABI v1

**Files:** new perf-core/native-abi plus turbo-quant-c and turbo-quant-zig integration

1. Add layout, version, caller-buffer, capacity, partial-failure, and allocator-ownership tests.
2. Implement versioned descriptors and status/error contracts.
3. Migrate C and Zig; then compile Mojo, Nim, and Go against the same headers.
4. Run sanitizers or equivalent memory checks plus concurrent round trips.
5. Review, commit, and snapshot.

## Task 6: Implement model-family kernel packages

**Files:** cohesive modules under perf-core/model-kernels, each below 350 lines where practical

1. Attention: GQA, MLA, CCA, paged/tree attention and compressed cache operations.
2. MoE: routing, grouping, expert GEMM, shared experts, and reduction.
3. Recurrent: DeltaNet, short convolution, scan, and recurrent state updates.
4. Diffusion: active-token denoise/remask scheduler and fused update candidates.
5. Quantized: ternary and sub-byte layout conformance plus fused operations.
6. For each package, add oracle tests before optimized implementations and shape-bucket benchmarks.

## Task 7: Integrate Metal runtime selection

**Files:** perf-core/metal-runtime and MLX integration points

1. Test bounded compilation, pipeline caching, buffer reuse, cancellation, and malformed kernels.
2. Implement device fingerprinting, candidate compilation, command scheduling, and telemetry.
3. Connect ModelPlan to registry selection and emit structured ExecutionTrace records.
4. Add policy that forbids untrusted runtime source in production mode.
5. Run correctness and performance gates, review, commit, and snapshot.

## Task 8: Build AX, DX, UX, and governance surfaces

**Files:** focused CLI/report modules and thin Python/Bun edges

1. Add CLI contract tests for inspect, explain, tune, replay, compare, and evidence export.
2. Implement concise errors containing model, operator, shape, backend, and violated contract.
3. Implement benchmark and quality dashboards with provenance and confidence intervals.
4. Add candidate promotion, quarantine, rollback-by-policy, and audit-trail workflows.
5. Review, commit, and snapshot.

## Task 9: Full-system acceptance

1. Run workspace tests across stable and nightly plus polyglot feature combinations.
2. Run ABI memory checks, fuzzing, concurrency, cancellation, and soak tests.
3. Execute Qwen coding/tool traces, NIAH, MMLU-Pro, GPQA, terminal-bench, and model-family
   conformance using stock and frontier reference models.
4. Lock regression baselines for latency, throughput, memory, energy, quality, and stability.
5. Produce requirement-by-requirement evidence, code review, coherent commits, and Airlock snapshot.
