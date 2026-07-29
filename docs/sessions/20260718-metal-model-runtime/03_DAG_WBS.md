# DAG and Work Breakdown

## Dependency Graph

    R0 evidence and red-test inventory
      -> R1 execution-plan domain
      -> R2 kernel registry and tuning store
      -> R3 native ABI v1
    R1 -> K1 attention and state kernels
    R1 -> K2 sparse MoE kernels
    R1 -> K3 recurrent and convolution kernels
    R1 -> K4 diffusion scheduler and kernels
    R1 -> K5 ternary and sub-byte kernels
    R2 + K1..K5 -> I1 runtime selection and observability
    R3 + K1..K5 -> I2 Zig, Mojo, C, Nim, and Go integration
    I1 + I2 -> V1 model-family conformance
    V1 -> V2 quality, performance, energy, and stability gates
    V2 -> G1 promotion governance and release evidence

## Critical Path

1. Repair the workspace test baseline and eval-harness public contract.
2. Add execution-plan types and reference interpreter with contract tests.
3. Add kernel registry, deterministic selector, tuning records, and trace schema.
4. Correct tree-attention and speculative-state semantics against scalar oracles.
5. Establish Native ABI v1 and migrate C/Zig first.
6. Implement and benchmark model-family kernel packages.
7. Run real model, agentic trace, NIAH, quality, and stability acceptance.

## Parallel Work Packages

| Lane | Scope | Dependency | Exit evidence |
|---|---|---|---|
| A | Eval harness correctness | R0 | Workspace green; deterministic loaders and scoring |
| B | Domain and registry | A baseline | Contract tests and serialized plans |
| C | Attention and speculation | B | Oracle parity and memory bounds |
| D | MoE, recurrent, diffusion | B | Family-specific conformance and benchmarks |
| E | ABI and polyglot | B | Sanitized cross-language round trips |
| F | AX, DX, UX, governance | B and registry | CLI reports, traces, promotion controls |
| G | Full acceptance | C through F | Reproducible regression bundle |

## Review Gates

Each implementation package receives specification review, code-quality review, targeted tests,
workspace tests, benchmark comparison, and Airlock snapshot before the next dependent package.

## Forward Kernel DAG (2026-07-19)

    MoE top-k router [complete]
      -> grouped expert GEMM [complete — model-kernels::moe::gemm_tiled, scalar-tile path, oracle parity pinned, kernel-registry candidate wired with tuning + coverage]
      -> weighted expert reduction [complete — model-kernels::moe::reduce_tiled, scalar-tile path, oracle parity pinned, kernel-registry candidate wired with tuning + coverage]
      -> end-to-end Qwen/OLMoE model run
      -> Step / Kimi K2 / GLM / OLMoE / MiniMax conformance

    KDA chunk scan
      -> recurrent-state ABI
      -> Falcon-H1 / Nemotron / Kimi Linear hybrid scheduling

    packed ternary GEMM
      -> BitNet b1.58 conformance
      -> Bonsai shared low-bit dispatch evidence

    MTP proposal tree
      -> batched verification
      -> Step / Nemotron / GLM agentic decode acceptance

    active-position compaction
      -> remasking scheduler
      -> Seed Diffusion / Mercury coding acceptance
Current critical path: end-to-end Qwen/OLMoE model run -> latency, memory,
energy, and quality regression baselines. The previous critical-path item
(weighted expert reduction) was completed at turn-13 with
`model-kernels::moe::reduce_tiled`, oracle parity pinned against the
scalar reference, a `weighted_reduce_moe` kernel-registry candidate
(scalar + tiled) wired into `coverage_matrix.rs` and the SOTA operator
suite, and a deterministic bench envelope (5 row contexts × 5 seeds = 25
rows) mirrored against the `grouped_gemm_moe` envelope so the two
families are directly comparable. The turn-12 critical-path item
(grouped expert GEMM) was completed at commit c735ea0 with
`model-kernels::moe::gemm_tiled`.

## Native artifact promotion sub-DAG (2026-07-29)

    checked-in MSL sources
      -> Xcode-beta AIR compilation (17/17)
      -> combined metal-runtime.metallib
      -> deterministic manifest + SHA-256 allowlist [current]
      -> current-HEAD immutable candidate envelope
      -> verified device load / dispatch
      -> Qwen3.5 family acceptance (throttled, authorized)

The manifest/allowlist step is now implemented and covered by focused Rust tests. The final
two nodes remain intentionally open: no device dispatch or model/evaluation workload is run in
the overload-safe lane.

Selector reachability now covers the Bonsai ternary path and grouped-MoE matmul path, so these
families can reach catalogued native source during reference compilation. Runtime execution is
still gated on verified artifact loading and live device dispatch.

The selector-to-function-name catalog is now explicit and tested for all currently catalogued
native sources (MoE router/dispatch, Bonsai ternary, diffusion confidence, attention, and
recurrent families). Unknown tags fail closed before a Metal lookup.

`NativeKernelBundle` now joins the manifest-approved artifact to this catalog, producing a
verified `(artifact, tag, function)` binding suitable for the eventual Metal command encoder.

The three highest-leverage native leaves now consume that binding at cache lookup time: Bonsai
ternary GEMM, MoE router/grouped GEMM, and diffusion confidence. This removes function-name
drift at the final pre-dispatch boundary.

## Forward family expansion

    diffusion confidence
      -> confidence trajectory state
      -> active-position compaction
      -> remask scheduler
      -> bounded block-diffusion self-verification

Status update 2026-07-29: active-position compaction and remask now have Rust oracle contracts,
checked-in MSL, source-catalog entries, and native tag-to-symbol bindings. The next dependency is
trajectory-state storage plus a real command-encoder dispatch path; no workload is launched until
the overload governor and explicit Qwen3.5 acceptance window are satisfied.

Status update 2026-07-29b: trajectory state now has a Rust oracle and catalogued Metal update
kernel. The remaining diffusion source path is command-encoder wiring, then bounded block
self-verification; source compilation is 20/20.

Status update 2026-07-29c: `StateKind::DiffusionTrajectory` now gives the plan layer an explicit
persistent slot for confidence/entropy/momentum/convergence metadata. Isolated validation now
passes (`state_kind_tag_for_each_variant`, 1/1); no runtime workload was started.

Status update 2026-07-29d: `metal-runtime::DiffusionStateLayout` now defines the mixed-dtype
allocation contract: three `f32` arrays plus mask/converged byte arrays, with checked arithmetic.
Focused layout tests pass 2/2; command-encoder binding remains the next runtime boundary.

Status update 2026-07-29e: `DiffusionDispatchPlan` now binds that layout to the ordered
`active_compact -> remask -> trajectory` stages and exposes the token-sized thread grid. Focused
dispatch-plan tests pass 2/2; it is a deterministic command-encoder input contract, not device
execution evidence.

Status update 2026-07-29f: feature-gated Metal bindings now expose all three catalogued stages
with strict shape checks, shared buffer construction, thread-grid dispatch, and command-buffer
status errors. `cargo check -p metal-runtime --features metal` passes; no device call was made.

Status update 2026-07-29g: the host parity oracle now compares compacted values/positions, masks,
and floating-point trajectory outputs with explicit shape and tolerance errors. Focused parity
tests pass 2/2; device parity remains intentionally uninvoked.

Status update 2026-07-29h: added an ignored, explicit-env Metal integration fixture covering all
three stages and the parity oracle. The test target compiles with `--features metal`; execution
requires an allowlisted artifact and is not part of ordinary CI.

Status update 2026-07-29i: added a host-only `DiffusionVerificationPlan` that partitions token
state into a finite block budget and validates each returned `f32` block through the existing
parity oracle. Three focused tests pass; this is a bounded command-encoder contract, not device
or Qwen3.5 execution evidence.

Status update 2026-07-29j: added `DiffusionDispatchTelemetry` / `DiffusionDispatchReport`, which
rejects invalid timing, stage order, and incomplete-without-error envelopes while deriving total
duration and fallback state. Three focused telemetry tests pass; no workload was launched.

Status update 2026-07-29i: hardened the diffusion dispatch boundary with a pure threshold
validator shared by remask and trajectory bindings. NaN/Inf and confidence values outside
`[0,1]` fail closed before Metal allocation; focused tests pass 2/2.

    MoE router/grouped GEMM
      -> top-1 vs top-2 load histogram envelope
      -> expert locality / grouped-GEMM tile sweep
      -> Qwen3.5/OLMoE/DeepSeek reference conformance

    Bonsai ternary GEMM
      -> group-scale and K-tail parity
      -> zero-elision / byte-alignment tile sweep
      -> quality/perplexity envelope before promotion
