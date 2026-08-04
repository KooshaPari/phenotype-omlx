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

## 2026-07-22 Evidence

- `model-kernels` MLA and MLA-cache unit tests: 7 passed.
- Metal DeltaNet, Zaya CCA, and MLA-cache parity tests: 1 passed each with their pinned
  metallib artifacts; missing artifact environment variables fail closed before dispatch.
- Recurrent percentile baseline is recorded in
  `research/baselines/metal-runtime-recurrent-percentiles-20260722.json`.
- The recurrent baseline now includes `retnet_retention_step_f32` (9 samples; median and p95)
  alongside DeltaNet, CCA, and MLA-cache.
- The public baseline is explicitly labeled `workload: binding-smoke`; it is a dispatch/ABI
  regression guard, not a claim of production-shape throughput.
- The recurrent baseline now also includes `mamba_selective_step_f32`, using the local
  model-kernels selective-scan convention and the artifact-backed Metal wrapper.
- A chunked `mamba_selective_scan_f32` Metal path now has an artifact-backed parity test against
  the multi-step Rust selective-scan oracle; state continuity is preserved across the sequence.
- The recurrent percentile baseline now measures both Mamba single-step and chunked-scan paths,
  making the optimization directly comparable before any production-shape tuning.
- A separate production-shape Mamba record is now captured at 256 steps x 64 state channels in
  `research/baselines/mamba-production-percentiles-20260722.json` (median/p95, artifact pinned).
- The production-shape Mamba scan now uses a 256-thread channel-parallel reduction while keeping
  timestep ordering sequential; the refreshed run measured 2356.875 us median / 4649.334 us p95.
- Boundary parity covers state dimensions 1, 255, 256, and 257, guarding the threadgroup stride
  and exact-256 boundary against state corruption.
- Pure shape validation is exported and tested independently; malformed inputs reject before
  artifact loading or Metal dispatch.
- Split-chunk continuity is covered: two sequential 2-step dispatches must match one fused
  4-step dispatch in both outputs and final state.
- Production comparison now records fused scan versus 256 repeated single-step dispatches:
  fused median 2473.459 us vs repeated median 203610.333 us (~82x lower end-to-end latency).
- Kernel-registry now exposes stable builders for the new runtime paths: `retnet_key` uses the
  existing `Recurrent` operator kind, while `mamba_scan_key` uses `Scan`; both preserve the
  existing serialized discriminants and are covered by unit contracts. Full kernel-registry
  validation passes: 33 unit, 14 contract, 20 governance, 10 fuzz, and 136 SOTA-operator tests.
- Selector-level integration now routes both builders to artifact-tagged Metal candidates;
  recurrent SOTA coverage passes 17 tests, including RetNet selection and Mamba scan selection,
  trace, experimental-policy, and dtype-rejection contracts.
- The intended environment integration now extends the installed MLX namespace with the
  persistent TurboKVCache layer; readiness reports `TurboKVCache ready (Metal)`. A live
  Qwen/Qwen3.5-0.8B in-process MLX smoke passed exact needle match and is pinned in
  `research/baselines/qwen35-mlx-live-20260722.json`. This is real runtime evidence, but not yet
  the required 8192-token workload; KV-cache compression metrics remain explicitly inapplicable
  to this Qwen3.5 architecture path.
- A real Qwen3.5 NIAH run at exactly 8192 tokens completed through `scripts/niah_benchmark.py`:
  after canonical control-token extraction, prefill 10,354 ms, decode 3,182 ms, 0.3 tok/s,
  exact needle match. The result is pinned in
  `research/baselines/qwen35-niah-8192-20260722-rerun.json`; the evaluator has focused tests for
  Qwen thinking/end-of-text suffixes and does not use substring matching to award exact credit.
- The freshly rebuilt artifact set passes the ignored recurrent Metal benchmark with all
  artifact variables configured. The public binding benchmark remains separately gated on its
  legacy ADALN artifact; the build script now emits `adaln_rms.metallib` and the artifact is
  hash-pinned, so the remaining public run is an execution-duration issue rather than a missing
  source artifact.
- The public artifact-backed binding benchmark now passes with the complete rebuilt set (including
  ADALN, Flow CFG, joint attention, RoPE-3D, and temporal attention): 9 kernel bindings executed,
  each reporting median/p95 latency, with 1 test passed and 0 failures.
- The plan/runtime bridge now carries an explicit `OperatorKind::MambaScan` variant and maps it to
  `KernelOp::MambaSelectiveScan`; model-plan serde/tag coverage and a dispatch integration test
  pass. This closes the prior gap where linear-recurrent Qwen hybrid plans fell through to no
  registry tag.
- TurboKVCache production metadata now reports `cache_applicability` and a reason when a model
  exposes no `TurboKVCacheLite` layers. Qwen3.5 acceptance tests treat that architecture-specific
  result as not applicable rather than falsely requiring ordinary KV compression.
- `MetalKernelBackend` now performs an in-process `mx.fast.metal_kernel` probe on the real request
  path and returns structured `custom_metal_probe` evidence. On the host, `mlx.core` reports Metal
  available and the probe returns `True`; the backend regression test verifies the evidence is
  surfaced alongside the preserved Qwen3.5 model path.
- Metal responses now include an architecture-derived `model_kernel_plan` (for example, Qwen3.5
  linear-attention layers are classified as DeltaNet-family) so registry coverage can be audited
  against the actual loaded model. The plan is explicitly marked `executed: false` until MLX layer
  hooks are replaced with verified custom-kernel calls; it is provenance, not an execution claim.
- The Bonsai ternary GEMM shader now decodes packed 2-bit weights byte-wise, keeping each packed
  byte in a register and eliminating per-weight division/address recomputation. The full shader
  build completes through `scripts/build_moe_metallibs.sh`; existing MoE parity tests remain the
  correctness gate for the shared artifact set.
- With `OUT_DIR` correctly exported, the complete MoE artifact parity suite passes 8/8, including
  grouped-GEMM assignment-oracle parity and top-k router parity. The diffusion confidence artifact
  parity test also passes 1/1 against stable argmax and softmax-max references.
- RetNet is now a first-class model-plan operator and kernel tag: `OperatorKind::RetNet` maps to
  `KernelOp::RetNet` and the Metal dispatch bridge test passes. Model-plan (60), model-kernels
  (203), and Metal dispatch (19 focused) tests all pass after the addition.
### Qwen3.5 native recurrent-kernel evidence (2026-07-23)

The installed MLX-LM Qwen3.5 path is not purely scalar: `GatedDeltaNet` invokes
`gated_delta_update`, whose production implementation builds and dispatches fused
`mx.fast.metal_kernel` kernels for gated-delta recurrence. This should be benchmarked
against phenotype-omlx's DeltaNet artifact, but must remain separately labeled as
`mlx_native` until a direct replacement hook is implemented.

Native gated-delta dispatch baseline: median 223.417 us / p95 298.542 us at
`B=1,T=8,Hk=2,Hv=4,Dk=32,Dv=16`. Add a dispatch-shape guard for `Dk < 32`; the upstream
kernel currently emits an invalid zero-length local array for that shape.
The native Metal kernel is numerically parity-checked against the compiled ops reference:
maximum output error `5.72e-6`, state error `0.0`, eight-token recurrence.

### Candidate rebind preparation (2026-08-03)

`scripts/tests/test_candidate_rebind.py` is a pure local contract suite. It creates temporary
Git repositories and synthetic, canonical JSON inputs; it does not load a model, contact Harbor,
or dispatch Metal. The suite verifies that preparation accepts only a clean current checkout with
live, exact-8192, no-retry/no-fallback Qwen3.5 evidence and compatible 20-shader Metal
provenance. It rejects a stale source head, Qwen2.5, a retried Harbor result, a dirty checkout,
historical candidate-manifest output, and any existing output path. Successful preparation still
emits `review_required` and `promotable=false`.

The regression matrix additionally rejects a noncanonical Qwen3.5-looking model ID, a non-NIAH
or multi-trial result, boolean values masquerading as numeric Harbor metrics, and missing
repository-relative artifacts. These tests protect the exact SSOT, bounded-window, and
content-addressed provenance requirements without executing a model or Harbor task.

### Candidate rebind P1/P2 regression coverage (2026-08-04)

The rebind suite replaces the Harbor-envelope path after that envelope has been parsed and
validated; the resulting record must retain the original validated input SHA-256, never the
replacement's digest. It independently rejects a multi-trial envelope and each mismatched
candidate repository or branch, while preserving the authorization-sidecar and artifact
descriptors in successful review records. These are filesystem-only contract tests and do not
start Harbor, MLX, Metal, or Qwen3.5.

Authorization-sidecar coverage also rejects a valid, content-addressed sidecar whose `window_id`
does not equal the envelope's window marker. This guards against combining unrelated authorization
and run records while keeping execution evidence and authorization distinct.

Fresh artifact-backed recurrent baseline (9 samples): DeltaNet median/p95 `448.292/708.0 us`,
CCA `317.833/459.542 us`, MLA cache `305.916/368.5 us`, RetNet `328.958/405.375 us`,
Mamba step `317.458/364.084 us`, and Mamba scan `345.125/462.083 us`.
