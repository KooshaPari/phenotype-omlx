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

Status update 2026-07-29k: bound telemetry construction to `DiffusionDispatchPlan`, rejecting
stale layouts or stage arrays before an envelope can be emitted. Five focused telemetry tests pass.
The ignored Metal fixture now compiles successfully with Xcode-beta and `--features metal`
(`--no-run`, isolated target); its device test remains unexecuted.

Status update 2026-07-29l: added `DiffusionStageTelemetry::from_result`, a shared conversion from
command success/error plus elapsed time into validated completion, error, and fallback fields.
Six focused telemetry tests pass; this remains a host-side policy primitive until the live encoder
is explicitly exercised.

Status update 2026-07-29m: added outcome-returning Metal entry points for active compaction,
remasking, and trajectory update. Each records elapsed time and preserves native failure in the
telemetry envelope rather than silently converting it to success. Host telemetry tests pass 6/6;
the Xcode-beta Metal fixture target compiles with `--no-run`; no command buffer was executed.

Status update 2026-07-29n: the ignored fixture now consumes those outcome APIs and aggregates all
three stage envelopes through the validated dispatch plan before parity checks. Added a bounded
`Promote`/`Fallback`/`Rollback` policy with explicit failed-stage limits. Seven focused telemetry
tests pass; fixture validation remains compile-only.

Status update 2026-07-29o: `DiffusionDispatchPlan::evaluate` now re-validates report layout and
stage order before applying the rollback policy. Five focused dispatch tests pass; promotion can no
longer be decided from telemetry belonging to a stale plan.

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

Status update 2026-07-30a: added the host-only `DiffusionDispatchPlan::evaluate_outcomes`
orchestration helper. It consumes typed active-compaction, remask, and trajectory outcomes,
retains their outputs, derives a plan-bound report, and returns the bounded `Promote`, `Fallback`,
or `Rollback` decision. Focused dispatch tests pass 7/7; no Metal, device, or Qwen3.5 workload
was executed.

Status update 2026-07-30b (P4 promotion gate): live promotion requires a fresh immutable
candidate envelope tied to the current branch and exact HEAD, with the manifest and every
`.metallib` SHA-256 recorded, the Xcode-beta/device fingerprint captured, and the Qwen3.5
model identifier, Harbor job/trial, requested and observed context lengths, prompt hash,
fallback/error counts, reward/pass@1, and oracle/result artifact hashes present. Existing
Harbor/candidate records reference older heads and remain review-only; no stale manifest or
prior successful trial may be re-used as current-HEAD evidence. The gate is held until the
current-HEAD envelope is emitted and receives final local promotion review; no workload was
run in this turn.

Status update 2026-07-30c (bounded artifact inventory): the current-head provenance envelope
(`artifacts/candidate-provenance-20260730.json`) and the stale historical `candidate-manifest.json`
were found under the session directory. A max-depth-six inventory of `phenotype-omlx/` and
`Downloads/` found no `.metallib` artifact available for allowlist verification. G6 therefore
remains held pending a fresh current-HEAD native artifact envelope; no source or workload action
was performed.

Status update 2026-07-31a (compile-only artifact reproduction):
`scripts/build_metal_runtime_bundle.sh` compiled 20 checked-in shaders with Xcode-beta into
`/tmp/phenotype-omlx-metal-current-20260731/metal-runtime.metallib` (111,965 bytes,
SHA-256 `ff53ce9e3d21244e4799887f72211133a4173c3671552555dfa7336bc7aa3d83`). The actual
repository HEAD was `ba30267b`; its only change after `f2127090` was provenance metadata, so
the compiled shader inputs are source-equivalent to the requested `f2127090` candidate. This
is compile-only evidence: device/runtime execution remains false, and promotion stays blocked.

Status update 2026-07-31b (research-to-experiment bridge): the next optimization wave is
ordered by evidence risk, not projected speedup. First lock exact Qwen3.5 state continuity
(model/tokenizer/config/kernel-plan provenance, canonical prompt ordering, position range, and
dtype); then measure output-KV promotion and content-addressed RAM/NVMe state tiers. Semantic
retrieval may propose a branch but cannot authorize KV reuse. JetSpec/DSpark-style speculation,
diffusion remask/trajectory scheduling, and ternary zero-elision remain separate experiments.

The bounded experiment matrix is:

| Node | Experiment | Required evidence | Promotion rule |
|---|---|---|---|
| R1 | exact prefix/KV continuation graph | hit/miss, new tokens, state movement, peak memory | no quality or provenance regression |
| R2 | output-KV promotion | authoritative decode marker and replay parity | exact replay only |
| R3 | RAM/NVMe state tiers | content hash, prefetch latency, eviction/rollback | bounded queue and no stale state |
| R4 | 3090 Ti primary / 1080 Ti drafter | per-device latency, memory, acceptance rate | drafter never authoritative |
| R5 | diffusion active compaction/remask/trajectory | stage order, resource fences, parity, quality | current-head device evidence required |
| R6 | Bonsai/BitNet ternary kernels | packed/unpacked parity, K-tail, scale/zero metadata | quality/perplexity envelope required |

All R-nodes inherit the overload governor: one bounded trial, fixed context, no automatic
retries, explicit timeout, and immutable result hashes. Apple Metal dispatches must publish
resource dependencies/barriers before encoder fusion is considered. Harbor/Portage task and
dataset artifacts are the system of record; ad-hoc scripts may only prepare or verify them.

## VRAM-note serving sub-DAG (2026-07-31)

    exact Qwen3.5 state replay
      -> hot 3090 Ti shared path/KV/GDN/hot experts
      -> measured 1080 Ti coarse stage or warm-expert residency
      -> DRAM/page-cache warm tier
      -> content-addressed NVMe cold catalog
      -> slack-bytes admission + prefetch/backpressure
      -> token-fate/state-hash envelope and promotion review

The note's actionable experiments are bounded to cold/warm cache, mmap versus concurrent
`pread`, kernel-ready layouts, expert reuse/prediction, and separate prefill/decode stage-share
sweeps. A full-KV or dense-weight copy per token is rejected by the design contract. Each trial
must report cache hit/miss, physical disk bytes, page faults, PCIe bytes, queueing, device role,
state hashes, and fallback/rollback; no hardware-modification or market claim can satisfy a
Qwen3.5 acceptance gate.

## Snapshot integrity gate update (2026-07-31)

    cache resolution + safe index paths
      -> safetensors payload accounting
      -> index metadata-scope reconciliation
      -> current-head provenance
      -> authorized Harbor/device window
      -> promotion review

The local Qwen3.5 snapshot is not treated as corrupt: `config.json` declares the vision shard as
a sidecar, and its base `model.safetensors` payload matches `metadata.total_size`. The verifier
records both filesystem and payload totals, hashes every indexed shard, and accepts only the
explicit `declared_sidecars_excluded` scope with a warning. No runtime window may bypass this
gate or substitute the older Harbor artifact.

## Current-head candidate reconciliation (2026-08-01)

    stale recovery manifest
      -> current branch/head + compile-only references
      -> direct review assertions 9/9
      -> bounded Qwen3.5 Harbor/device window
      -> benchmark envelope + promotion review

The candidate manifest now reports `blocked` rather than inheriting historical live-run claims.
This is an evidence correction, not a runtime result; no model or device workload was launched.

The current-head Metal compile manifest is now present in the candidate record with shader count,
metallib SHA-256, manifest SHA-256, and `verified_compile_only` status. It must not be promoted to
device evidence without an authorized bounded execution.

## Evidence-attestation prerequisite (2026-08-05)

    clean Portage Git root
      -> signed immutable Harbor/Hub envelope from an upstream issuer
      -> strict OMLX trusted-envelope verification
      -> current-head Qwen3.5 bounded execution evidence
      -> independent promotion review

The OMLX verifier and its fixture-only Ed25519 policy are ready to reject malformed or unsigned
evidence, but Portage/Hub does not yet publish a signer-backed export. The trusted verifier is
therefore a prerequisite enforcer, not a route around that missing upstream issuer. A future
window must use a conflict-free Portage root, currently the hygiene candidate
`worktrees/portage/fix-langsmith-importerror`, and must remain subject to the one-trial overload
governor.

## Diffusion boundary finiteness gate (2026-08-05)

    host trajectory oracle finiteness rules
      -> shared Metal-boundary value validator
      -> remask/trajectory dispatch rejection before pipeline lookup
      -> bounded device fixture

The Metal dispatch boundary now rejects non-finite confidence and non-finite or negative entropy
before allocation or catalogued pipeline lookup. This aligns the device boundary with the host
trajectory oracle; it is host-only contract evidence and not device or Qwen3.5 execution evidence.
