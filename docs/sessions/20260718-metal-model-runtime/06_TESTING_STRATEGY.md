# Testing Strategy

## Correctness Layers

1. Domain tests validate plans, state transitions, shape rules, serialization, and rejection.
2. Reference-oracle tests compare every optimized kernel across edge and randomized dimensions.
3. ABI tests exercise ownership, capacity, failure cleanup, nullability, concurrency, and versioning.
4. Model conformance tests execute representative layers and short end-to-end generations.
5. Trace replay tests preserve real agentic, MoE, diffusion, recurrent, and quantized workloads.

## Model Acceptance

- Qwen: coding/tool traces, long-context retrieval, sparse experts, DeltaNet or GQA state.
- DeepSeek: MLA cache equivalence, routed experts, proposal verification.
- LFM and ZAYA: convolution or compact-attention schedules and expert semantics.
- Bonsai: exact packed bytes, dequantization tolerance, fused-operation parity.
- Diffusion: denoise/remask invariants and convergence under changing active-token masks.
- Mamba, Jamba, RWKV: chunking invariance, recurrent-state continuity, bounded memory.

## Quality Evaluation

Use no more than ten complementary suites: MMLU-Pro, GPQA Diamond, terminal-bench, SWE-bench or a
locally licensed coding equivalent, BFCL or tool-use traces, NIAH/long-context retrieval, HumanEval+
or LiveCodeBench, instruction following, perplexity/calibration, and model-family conformance.
Dataset loaders record source revision and split. Scoring is deterministic where possible and judge
models are versioned where unavoidable.

## Performance and Stability

Record warmup, sample count, median, p95, p99, variance, tokens/s, time-to-first-token, peak memory,
allocated bytes, dispatch count, compile time, cache hit rate, and energy where available. Include
soak, repeated load/unload, concurrent requests, cancellation, malformed input, and memory-pressure
tests. Baselines are keyed by hardware, OS, compiler, model, plan, and source revision.

## Required Gates

- Targeted tests pass before widening scope.
- Workspace tests and all feature combinations compile.
- Sanitizers or equivalent ABI memory checks pass.
- Quality is non-regressing within declared confidence and tolerance.
- Performance claims include reproducible before/after artifacts.
- No edited module exceeds 500 lines; new modules target 350 lines or fewer.
