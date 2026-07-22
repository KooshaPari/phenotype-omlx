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
