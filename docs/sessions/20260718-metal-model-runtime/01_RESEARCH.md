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
