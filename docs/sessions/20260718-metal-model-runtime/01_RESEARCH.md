# Research

## Primary Sources

- Qwen3-Coder-Next: https://huggingface.co/Qwen/Qwen3-Coder-Next
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- LLaDA: https://arxiv.org/abs/2502.09992
- Dream: https://arxiv.org/abs/2508.15487
- Mamba: https://arxiv.org/abs/2312.00752
- Jamba: https://arxiv.org/abs/2403.19887
- RWKV: https://arxiv.org/abs/2305.13048
- MLX custom kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- Bonsai model family: https://huggingface.co/microsoft
- Liquid Foundation Models: https://huggingface.co/LiquidAI

## Verified Operator Families

| Family | Distinct runtime requirements |
|---|---|
| Qwen3-Coder-Next | Hybrid DeltaNet, GQA, sparse MoE, long-context agentic decode |
| DeepSeek | MLA cache compression, routed MoE, multi-token prediction |
| LFM2 | Gated short convolution, sparse GQA, sparse MoE |
| ZAYA | Compressed context attention, top-1 nonlinear MoE, compact state |
| Bonsai | Ternary group quantization with fixed packed block contracts |
| LLaDA and Dream | Parallel masked denoising, remasking, variable active-token sets |
| Mamba, Jamba, RWKV | Recurrent state update, scan kernels, hybrid attention scheduling |

## MLX and Metal Findings

MLX custom Metal kernels expose specialization dimensions but do not provide a complete public
autotuner. Therefore this runtime must own candidate generation, warmup, measurement, correctness
validation, hardware fingerprinting, cache invalidation, and tuning records. Dynamic compilation
must be bounded and cached; production execution cannot silently compile arbitrary source.

## Architectural Inferences

- Model identity is insufficient for dispatch; selection keys must include operator shape, dtype,
  quantization, state layout, device features, and correctness policy.
- MoE needs fused routing, grouping, expert GEMM, and reduction candidates rather than a generic
  dense-GEMM loop.
- Diffusion needs an explicit scheduler and active-token mask, not an autoregressive compatibility
  wrapper.
- Ternary formats require exact byte-level conformance tests before performance work.
- Cross-language implementations should share buffers and ownership rules, not duplicate domain
  models or convert tensor objects at every boundary.

## Research Gaps to Keep Live

- Measure practical Metal sparse-MoE fusion limits by expert count and token batch.
- Compare MLA and CCA cache layouts against Apple GPU cache-line behavior.
- Evaluate recent block-diffusion, recurrent-hybrid, and sub-byte quantization papers as they land.
- Verify model configuration contracts from released checkpoints before adding a family preset.

## 2026-07-19 Frontier Expansion

Primary-source review added twelve high-leverage families to the forward kernel map: Kimi Linear
(KDA), BitNet b1.58, Step-3.5-Flash, Seed Diffusion, Falcon-H1, Nemotron 3 Hybrid, Kimi K2/K2.5,
MiniMax-M1, xLSTM, GLM-4.5/4.7, OLMoE, and Mercury Coder. The resulting implementation order is:

1. KDA chunk scan and recurrent-state ABI.
2. Native packed ternary GEMM for BitNet.
3. Grouped expert GEMM and weighted reduction after the completed top-k router.
4. MTP proposal and batched verification.
5. Active-position compaction and remasking for diffusion decoders.

The first real Metal kernel is now the MoE top-k router. It uses stable score-descending and
expert-id-ascending ties, selected softmax, a scalar model-kernels oracle, and an artifact-only
production path. `MetallibArtifact` is an unforgeable capability outside the crate: only the
allowlist/hash-verifying loader can construct one.

## 2026-07-22 Metal Reuse and Model-Family Refresh

Apple's Metal best-practices guidance explicitly recommends creating persistent objects early and
reusing them, including one command queue per GPU; pipeline-state creation is an expensive GPU
state evaluation. This directly supports the next optimization: cache verified libraries,
compute pipeline states, and command queues instead of recreating them per binding call.

- Persistent objects: https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/PersistentObjects.html
- Command structure: https://developer.apple.com/documentation/Metal/setting-up-a-command-structure
- Metal 4 command queues and reusable argument tables:
  https://msc-kobol-public-prod.apple.com/documentation/Metal/understanding-the-metal-4-core-api

The current model-family map should include Qwen3.5/3.6 MoE and Qwen3.5-Omni hybrid
attention+MoE, DeepSeek MLA+MoE, Kimi Linear/KDA, LFM2 recurrent-convolution hybrids, ZAYA
compressed-context/top-1 MoE, BitNet/Bonsai ternary, and diffusion families such as Flux,
Wan, Qwen-Image, and masked diffusion decoders. Qwen3.5/3.6 share the same Transformers model
type, while Qwen3.5-Omni uses a hybrid attention MoE for its Thinker/Talker paths.

- Qwen3.5 MoE docs: https://huggingface.co/docs/transformers/main/model_doc/qwen3_5_moe
- Qwen3.5-Omni report: https://arxiv.org/abs/2604.15804
- BitNet official repository: https://github.com/microsoft/BitNet
- BitNet.cpp paper: https://arxiv.org/abs/2502.11880
- MLX Swift model registry (including Qwen MoE and BitNet entries):
  https://github.com/ml-explore/mlx-swift-lm/blob/main/skills/mlx-swift-lm/references/supported-models.md

Decision: benchmark and optimize shared Metal execution infrastructure before adding another
family-specific shader. The six-kernel artifact-backed baseline now records median/p95 and exact
artifact hashes, allowing setup-reuse changes to be measured without conflating model-family
correctness with compiler variance.

## ToMoE, diffusion, ternary, and agent-serving extensions (2026-07-22)

- [ToMoE](https://arxiv.org/html/2501.15316v1) motivates deterministic top-k/MLP top-1
  routing, fixed-capacity buckets, active-budget telemetry, and explicit overflow/drop
  accounting for dense-to-MoE conversion experiments.
- [EPS-MoE](https://arxiv.org/abs/2410.12247), [expert sharding](https://arxiv.org/abs/2503.08467),
  and [Speculative MoE](https://arxiv.org/abs/2503.04398) motivate load-based dense-vs-grouped
  dispatch, shard-aware tiles, and next-expert prefetch hints.
- [TerDiT](https://arxiv.org/abs/2405.14854), the [diffusion quantization survey](https://arxiv.org/abs/2505.05215),
  and [BitNet.cpp](https://arxiv.org/abs/2502.11880) motivate packed {-1,0,+1} arithmetic,
  zero-elision, fused scales, and per-timestep calibration/error reporting.
- [INFERCEPT](https://arxiv.org/abs/2402.01869) and [Preble](https://arxiv.org/abs/2407.00023)
  motivate pause/resume tool-call traces, KV tiering, prefix-cache hit metrics, and queue
  fairness in agent-serving benchmarks.

Implementation consequence: persistent Metal pipeline caching now covers the MoE grouped-GEMM
and top-k paths. Assignment indices are validated before GPU dispatch; empty assignments return
without issuing a zero-thread dispatch. Next tranche: capacity-aware routing, dense/grouped
selection, timestep-aware diffusion calibration, and packed ternary/W4A8 kernels.

## Additional 2025-2026 kernel candidates (2026-07-22)

- [TriRun / Spectra 1.1](https://aclanthology.org/2025.acl-long.1294/) reports packed low-bit
  inference kernels; this is a direct follow-up for the Bonsai ternary path and motivates
  comparing bit-plane packing against the current scalar unpack loop.
- [LLaDA-MoE](https://arxiv.org/abs/2509.24389) combines masked diffusion with sparse MoE;
  prioritize a fused confidence/remask plus capacity-aware expert dispatch benchmark.
- [dInfer](https://arxiv.org/abs/2510.08666) separates diffusion iteration, decoding, and KV
  cache management; use those boundaries to benchmark Metal pipeline reuse instead of treating
  each denoise step as an isolated launch.
- [TC-MoE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bda8f7ac4c3ccc494b5206ee3fd92771-Abstract-Conference.html)
  suggests ternary expert-choice routing, a candidate for a fused route-and-packed-GEMM kernel.
- [Diff-MoE](https://proceedings.mlr.press/v267/cheng25d.html) and
  [Dense2MoE](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Dense2MoE_Restructuring_Diffusion_Transformer_to_MoE_for_Efficient_Text-to-Image_Generation_ICCV_2025_paper.html)
  motivate timestep-aware expert masks and spatial/token locality metrics.
- [RetNet survey](https://aclanthology.org/2026.findings-acl.256/) reinforces RetNet as a
  candidate recurrent/retention family; after DeltaNet parity, evaluate retention-state update
  and chunk-parallel kernels against RWKV/Mamba baselines.
- [LLaDA](https://arxiv.org/abs/2502.09992) and [DiffusionGemma](https://deepmind.google/models/gemma/diffusiongemma/)
  make parallel denoise/remask a first-class workload: fuse confidence extraction, active-mask
  compaction, and remask scheduling while recording denoise-step count and quality parity.
- [Scaling Laws and Efficient Inference for Ternary Language Models](https://aclanthology.org/2025.acl-long.1294/)
  and the released [Ternary Bonsai 8B](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-mlx-2bit)
  support a dedicated {-1,0,+1} bit-plane kernel with group scales, zero-elision, and packed
  accumulation rather than routing ternary weights through ordinary int8 GEMM.
- [BaseRT](https://huggingface.co/blog/basecompute/basert-release) is a relevant Apple-Silicon
  native-Metal baseline: compare launch overhead, tile shape, memory bandwidth, and small-batch
  decode against MLX-native dispatch before claiming custom-kernel wins.
- [MoE-Lens](https://huggingface.co/papers/2504.09345) and [SliceMoE](https://doi.org/10.18653/v1/2025.emnlp-main.807)
  motivate capacity-aware routing, expert-load histograms, and slice-level grouped-GEMM metrics;
  the existing MoE artifacts should add those measurements before model promotion.
### Qwen3.5 native MLX Metal path (2026-07-23)

The installed `mlx-lm` Qwen3.5 implementation (`models/qwen3_5.py`) routes linear
layers through `GatedDeltaNet`, which calls `gated_delta_update`. Its implementation
in `models/gated_delta.py` constructs `mx.fast.metal_kernel` variants
(`gated_delta_step`, masked, and vectorized forms) and selects them when the model is
not training. This is authoritative evidence that Qwen3.5 already has a native fused
Metal recurrent kernel in the MLX runtime.

This does **not** prove that phenotype-omlx's separate `deltanet_step.metal` or
`mamba_selective_scan.metal` artifacts replace the MLX implementation. Runtime
provenance therefore keeps those artifacts at `custom_kernel_execution_verified=false`
until an explicit layer hook dispatches them and records a counter.

Synthetic native-kernel benchmark on 2026-07-23 measured median 223.417 us and p95
298.542 us for `[B=1,T=8,Hk=2,Hv=4,Dk=32,Dv=16]`. A smaller synthetic `Dk=16`
failed upstream Metal compilation because its `Dk / 32` tile count becomes zero;
kernel selection must enforce `Dk >= 32` (Qwen3.5 production dimensions satisfy this).
Against MLX's compiled ops reference on the same tensors, the native kernel measured
`max_abs_error=5.72e-6` and `state_max_abs_error=0.0` across eight recurrent tokens.
### 2026-07-23 — Qwen3.5 gated-delta replacement

The isolated MLX-LM 0.31.2 runtime exposes `mlx_lm.models.gated_delta.gated_delta_kernel` as a
vector-gated `mx.fast.metal_kernel` with the shape contract `[B,T,Hk,Dk]`, `[B,T,Hv,Dv]`, and
`Dk % 32 == 0`. A real Qwen3.5-0.8B-OptiQ-4bit generation observed 54 native gated-delta
dispatches. `python/omlx_research/backends/qwen_gated_delta_kernel.py` now mirrors that kernel
behind an opt-in replacement and records dispatch/fallback counts; promotion still requires a
clean native-vs-custom parity run.

### 2026-07-27 - exact Harbor NIAH generation contract

Qwen's deployment guidance documents `chat_template_kwargs: {"enable_thinking": false}` as the
hard switch for direct responses, and the Qwen3.5 model card explicitly says `/think` and
`/nothink` are not the supported control surface. The Harbor smoke therefore sends the hard
switch and records `thinking_enabled=false` in its oracle envelope. With mlx-lm 0.31.3 this
switch adds 27 chat-template tokens; the prompt builder subtracts that measured overhead so the
API reports exactly 8192 prompt tokens. Live Apple Container evidence is recorded in
`artifacts/harbor-qwen35-20260727-8192.json`: reward 1.0, exact needle match, no errors or
retries, and `context_tokens_exact=true`.

References: https://github.com/QwenLM/Qwen3/blob/main/docs/source/deployment/vllm.md and
https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4.

### 2026-07-28 - MoE, ternary, and non-Qwen reference gap matrix

The current `model-kernels` and `metal-runtime` layers have scalar contracts plus optional
Metal implementations for top-k routing, assignment-list grouped GEMM, and packed 2-bit
ternary GEMM. The following references sharpen the optimization target without changing the
Qwen3.5-only production model policy:

| Family/reference | Useful systems implication | Current coverage | Gap / next experiment |
|---|---|---|---|
| [ToMoE](https://arxiv.org/html/2501.15316v1) | Fixed-budget top-1 MLP routing, deterministic structural selection, and load regularization | top-k router + capacity buckets | expose route/load histograms and benchmark top-1 dispatch separately from top-2 |
| [LFM2](https://arxiv.org/abs/2511.23404) | Hardware-in-the-loop hybrid short-convolution/GQA and an 8.3B/1.5B-active MoE reference | recurrent and MoE kernel families exist | add decode-shaped grouped-GEMM measurements at one-token and short-prefill regimes |
| [ZAYA1-8B](https://arxiv.org/abs/2605.05365) | 700M-active/8B-total MoE with bounded recurrent test-time state | ZAYA and LFM contracts/tests exist | keep bounded-state routing and expert-load telemetry separate from dense Qwen3.5 evidence |
| [Ternary Bonsai 8B](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-mlx-2bit) | 2-bit packed ternary weights with MLX/Metal deployment | packed CPU and Metal GEMM paths exist | verify group-scale layout parity; benchmark zero-elision and byte-aligned K tails |
| [Scaling Laws and Efficient Inference for Ternary LMs](https://aclanthology.org/2025.acl-long.1294/) | Ternary quality/performance trade-offs must be measured, not inferred from byte reduction | byte-level pack/unpack tests | add quality/perplexity envelopes before promoting ternary kernels |

Hardening decision: `router_topk` now rejects NaN and +/-infinity before sorting or softmax,
matching the Metal facade's finite-logit contract. This prevents non-finite weights from
reaching grouped GEMM and is covered by a regression test. No benchmark or model-quality claim
is made from this contract hardening alone.

### 2026-07-28 - MLA and agentic decode kernel targets

DeepSeek's official V2 implementation and FlashInfer's MLA API both treat latent-cache
attention as a distinct path rather than dense GQA. The hardware-centric MLA study further
motivates minimizing cache traversals and keeping latent/RoPE dimensions explicit:

- https://github.com/deepseek-ai/DeepSeek-V2
- https://docs.flashinfer.ai/api/attention.html
- https://arxiv.org/abs/2506.02523

The MLA Metal shader now uses a numerically stable online log-sum-exp recurrence. It computes
each cache-entry score once, updates the running maximum/norm, and accumulates values in the
same pass. A source contract test pins the one-traversal invariant. This is a kernel-level
optimization only: native Metal compilation, device parity, and Qwen3.5 end-to-end dispatch
remain separate evidence gates and must not be inferred from the source test.

### 2026-07-29 - diffusion decoding acceleration targets

Recent diffusion-language-model work changes the kernel target from a single argmax pass to a
stateful denoising scheduler:

| Reference | Kernel implication | Planned contract |
|---|---|---|
| [TSPD + Confidence Extrapolation](https://arxiv.org/abs/2605.30753) | Track per-token confidence, entropy, momentum, position, and convergence state | fused confidence/entropy reduction plus active-position compaction; preserve uncertainty metadata |
| [S2D2](https://arxiv.org/abs/2603.25702) | Mix block-parallel diffusion proposals with autoregressive self-verification | remask/proposal buffers and bounded verifier dispatch, sharing the existing speculative governor |
| [Not All Denoising Steps Are Equal](https://arxiv.org/abs/2604.02340) | Early/late steps can use a smaller denoiser while middle steps retain full capacity | step-class schedule in `StatePlan`; benchmark FLOP reduction separately from quality |
| [Discrete Diffusion Survey](https://arxiv.org/abs/2506.13759) | Masked diffusion requires active-token masks and repeated full-sequence attention | keep remask, mask compaction, and denoise logits as separate Metal kernels |

Implementation consequence: the existing `diffusion_argmax_confidence_f32` kernel is a useful
leaf, but it is not a complete diffusion runtime. The next source-level additions should be
confidence trajectory state, active-position compaction, and remask scheduling; none should be
promoted from source tests without live parity and quality envelopes.

### 2026-07-29 - active-position and remask source contracts

The first two scheduler leaves are now concrete: Rust `active_positions`/`compact_active`
preserve ascending scatter indices and reject value/mask shape mismatches, while Metal provides
`diffusion_active_compact_u32` and `diffusion_remask_confidence_f32`. Both are catalogued by
stable tag and concrete function symbol. The Xcode-beta source bundle compiles 19/19 shaders;
this remains source/artifact evidence, not device or Qwen3.5 quality evidence.

Trajectory state is now concrete as well: Rust tracks confidence, entropy, per-position
confidence momentum, decode step, and convergence; the Metal leaf is
`diffusion_trajectory_update_f32`. Focused oracle tests pass and the source bundle compiles
20/20 shaders. The contract intentionally keeps trajectory metadata separate from token values
so active compaction can scatter updates without losing uncertainty history.

### 2026-07-31 - state continuity, heterogeneous memory, and governed evaluation

The SSD/HW Stream research makes exact state continuity the first optimization gate: bind
every reusable KV/state block to model revision, tokenizer revision, canonical prompt order,
position range, dtype, and kernel-plan version. Output KV should be promoted only after an
authoritative decode, while semantic retrieval remains a proposer and never substitutes for
exact KV provenance. Recommended storage is content-addressed state blocks with RAM as the hot
tier and NVMe as a bounded cold/prefetch tier; cross-device transfers should carry IDs and
compact projections rather than unrestricted tensors.

Hardware roles remain asymmetric: the RTX 3090 Ti is the primary high-memory execution target;
the GTX 1080 Ti is drafter-only/low-priority unless an experiment proves otherwise. Each run
must record token fate (compute, cache hit/miss, state movement, and critical-path delay),
prefill amplification, speculative acceptance, novel-edge acceptance, peak memory, and
fallback/rollback counts. A governor must cap concurrency, queue depth, context length, and
retry count before any live experiment.

Primary references: MLX-LM model loading and prompt-cache guidance
(https://github.com/ml-explore/mlx-lm), Apple Metal compute/resource synchronization
(https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/ and
https://developer.apple.com/documentation/metal/resource-synchronization), Harbor's eval
and adapter contracts (https://www.harborframework.com/docs/run-jobs/run-evals and
https://www.harborframework.com/docs/datasets/adapters), BitNet b1.58
(https://arxiv.org/abs/2402.17764), KVQuant (https://arxiv.org/abs/2401.18079), TurboQuant
(https://arxiv.org/abs/2504.19874), and LLaDA diffusion language modeling
(https://arxiv.org/abs/2502.09992). These references guide experiments only; acceptance
remains tied to the exact local Qwen3.5 snapshot and immutable Harbor/Portage evidence.

### 2026-07-31 - VRAM note: tiered state and Qwen3.5 serving boundaries

The supplied 3090 Ti VRAM note rules out treating a consumer-board memory mod as the runtime
plan: capacity changes are controller/firmware/layout problems, not a safe path to promotion.
For Qwen3.5, use a hierarchical object fabric instead. Keep the shared dense path, router,
attention, KV/GDN state, and hot experts resident on the 3090 Ti; assign only measured coarse
stages or permanently resident warm experts to the 1080 Ti. Do not copy dense weights or a full
active KV to either GPU every token.

The actionable cache policy is: NVMe is the cold content-addressed catalog, host DRAM/page cache
is the warm tier, and GPU memory is the hot working set. Compare cold versus warm cache, mmap
versus concurrent `pread`, and kernel-ready versus transformed layouts; record physical disk
bytes/page faults rather than treating a warm page-cache hit as NVMe bandwidth. For MoE, trace
expert IDs, reuse distance, union size, and prediction accuracy; use protected hot residency,
probationary prefill entries, activation-aware prefetch, and grouped activation transfer. KV and
recurrent state instead require session hibernate/thaw, prefix reuse, and persistent ownership.

Admission should use a slack-bytes check (`object_size <= measured_path_bandwidth * deadline_slack`)
after queueing and contention. Separate prefill/decode policies and sweep 3090/1080 stage shares
only after exact state replay is green. These findings are design inputs, not Qwen3.5 evidence;
the acceptance model remains Qwen3.5-only and must report token fate, state hashes, cache tier,
device role, and fallback/rollback.

Sources: [NVIDIA RTX 3090 Ti specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/),
[vLLM KV/offload documentation](https://docs.vllm.ai/en/latest/), [LMCache](https://github.com/LMCache/LMCache),
[MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity), and
[LLM in a Flash](https://arxiv.org/abs/2312.11514). The note's market/mod claims are not
promotion evidence and are intentionally excluded from the Qwen3.5 gate.

### 2026-08-03 - immutable current-head evidence rebind preparation

The historical candidate manifest cannot be refreshed by editing its claimed source head: its
canonical digest, source compatibility, and runtime-evidence booleans are independent gates.
The Harbor envelope and Metal compile provenance must each bind to the exact current full Git
HEAD. A preparation artifact therefore records only validated input identities and SHA-256
digests, and terminates at `promotion.verdict=review_required`; it never marks evidence complete
or accepts promotion. This keeps the final local promotion review as a separate authority and
prevents a compile-only artifact from being mistaken for a workload result.

The preparer resolves the exact readiness model from `config/smoke_models.json` rather than
matching a Qwen3.5 substring. It also requires the bounded `omlx/niah-api-smoke` one-trial
contract, an authorization window plus sidecar digest, a clean branch, and locally present,
repository-relative Harbor artifacts whose bytes match the envelope SHA-256 values.

### 2026-08-04 - P1/P2 immutable input pinning

P1: validating JSON and later re-reading its path creates a time-of-check/time-of-use gap: a
replacement can make a review record name a digest that was never validated. The preparer now
opens each Harbor and Metal input once as a non-symlink regular file, validates the parsed bytes,
and records the SHA-256 from that same in-memory snapshot. P2: the review record preserves the
validated input descriptors as well as the authorization sidecar and every Harbor artifact
descriptor, allowing an independent reviewer to distinguish validated bytes from a later path
replacement. This is local provenance hardening only; it does not create workload evidence.

The authorization sidecar must additionally name the same `window_id` as the Harbor envelope.
This matches the canonical Harbor authorization record without inventing a separate `approved`
flag or treating a window marker as proof that a workload ran.

Repository-relative descriptor reads are now anchored at an opened repository-root descriptor.
The root requests `O_NOFOLLOW_ANY` where the platform exposes it, while every relative component
is opened with `O_NOFOLLOW` from the already-open parent descriptor; the terminal descriptor must
be a regular file. This rejects root, leaf, and intermediate symlink traversal without resolving
and reopening a string path. It does not address the separate Git-root identity TOCTOU between
repository discovery and Git metadata checks.

### 2026-08-05 - Harbor root hygiene and attestation boundary

The Harbor launcher now rejects a `PORTAGE_ROOT` that is not a Git checkout with a valid `HEAD`,
or whose tracked `HEAD` contains an unresolved merge-conflict marker, before Apple Container
preflight or the Harbor CLI can run. This prevents the launcher from using the conflicted
canonical Portage checkout as an accidental execution root. A separate clean Harbor-Langfuse
worktree is only a future hygiene candidate; it is not execution authorization or evidence.

Portage's local job, trial, task-lock, archive, and Hub records support cross-record consistency,
but the examined code has no result-artifact signature or server-issued provenance envelope.
Canonical JSON and SHA-256 protect an asserted byte sequence, not who executed it. Promotion-grade
evidence therefore needs an upstream Portage/Hub signer over immutable artifact identities. OMLX
can consume such an envelope fail-closed, but must reject unsigned Harbor output rather than
misrepresent local consistency as attestation.

The OMLX consumer boundary is `evals/harbor/interchange/trusted.py`, deliberately before the
permissive generic interchange loader and aggregate synthesizer. It accepts only the fixed
`trusted-harbor-envelope/v1` shape, an injected Ed25519 public-key policy, canonical signed
payload bytes, exact Qwen3.5 model/config/task/environment/context bindings, Harbor-to-Langfuse
identifier binding, UTC run ordering, and `harbor://` immutable identifiers. Result identifiers
must bind the Harbor job/trial, while each artifact carries a non-negative byte count and SHA-256.
The in-repository key is a deterministic test fixture only. Until Portage or Hub publishes a real
issuer/key policy and immutable signed envelope, all actual Harbor output remains untrusted for
promotion.

The trusted consumer now owns a separate file-ingress boundary rather than inheriting the generic
interchange loader's permissive path read. It opens the candidate envelope once with no-follow
semantics, requires a regular file after descriptor inspection, decodes UTF-8, rejects duplicate
keys and non-finite JSON constants, then verifies that same in-memory document. This closes a
path-substitution and parser-ambiguity gap without claiming that local bytes are issuer-attested.
