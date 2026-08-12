# V1 Model-Family Conformance Matrix (BACKLOG-OMLX-003)

This directory tracks the conformance status of every kernel family shipped
in `perf-core/model-kernels/` against the V1 promotion gate. The V1 gate
requires five independent signals:

1. **Real model** — kernel runs against an actual model-family checkpoint
   (not just synthetic shapes).
2. **Agentic trace** — a non-trivial multi-turn trace through a real agent
   (e.g., portage computer_1, terminus-2) with the kernel in the hot path.
3. **NIAH oracle** — needle-in-haystack reward ≥ 1.0 at both 8k and 32k
   context lengths (mirrors portage harbor-gate).
4. **Quality signal** — a downstream benchmark (MMLU / GPQA / perplexity)
   is within `±2σ` of the family baseline.
5. **Stability** — 50 consecutive runs across `n_concurrent` workers
   produce the same reward distribution within `1e-3` KL divergence.

## Status legend

- 🟢 **GREEN** — all 5 signals pass.
- 🟡 **AMBER** — at least one signal in flight, no blockers.
- 🔴 **RED** — known blocker; see `blockers:` column.
- ⚪ **N/A** — kernel family not yet started (no oracle).

## Matrix (as of 2026-08-10)

| Family                          | Module                       | Oracle                        | Real Model                   | Agentic Trace | NIAH    | Quality | Stability | Blockers                             |
| ------------------------------- | ---------------------------- | ----------------------------- | ---------------------------- | ------------- | ------- | ------- | --------- | ------------------------------------ |
| **Attention — Dense**           | `attention::dense`           | `dense_attention`             | ⚪                           | �             | ⚪      | ⚪      | ⚪        | —                                    |
| **Attention — GQA**             | `attention::gqa`             | `gqa_attention`               | 🟡 (Qwen3.5)                 | ⚪            | 🟢 (8k) | ⚪      | ⚪        | needs 32k NIAH                       |
| **Attention — MLA**             | `attention::mla`             | `mla_attention`               | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | no DeepSeek checkpoint yet           |
| **Attention — CCA**             | `attention::cca`             | `cca_attention`               | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | no CCA model family in V1 scope      |
| **Attention — CCA-block**       | `attention::cca_block`       | `cca_block_attend_oracle`     | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | no CCA model family in V1 scope      |
| **Attention — Paged**           | `attention::paged`           | `paged_attention`             | �                            | ⚪            | ⚪      | ⚪      | ⚪        | needs vLLM-style block-table fixture |
| **Attention — Tree**            | `attention::tree`            | `tree_attention`              | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | speculative-decode only              |
| **Attention — Sliding Window**  | `attention::sliding_window`  | `sliding_window_attention`    | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | Mistral-family not in V1             |
| **Attention — MLA Cache**       | `attention::mla_cache`       | `mla_cache_*`                 | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | tied to MLA model family             |
| **MoE — Router**                | `moe::router`                | `router_oracle`               | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | —                                    |
| **MoE — Dispatch**              | `moe::dispatch`              | `dispatch_oracle`             | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | —                                    |
| **MoE — Grouped GEMM**          | `moe::grouped_gemm`          | `grouped_gemm_oracle`         | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | —                                    |
| **MoE — Shared Experts**        | `moe::shared_expert`         | `shared_expert_oracle`        | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | —                                    |
| **MoE — Reduction**             | `moe::reduction`             | `reduction_oracle`            | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | —                                    |
| **Recurrent — DeltaNet**        | `recurrent::deltanet`        | `deltanet_oracle`             | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | —                                    |
| **Recurrent — Short Conv**      | `recurrent::short_conv`      | `short_conv_oracle`           | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | —                                    |
| **Recurrent — Mamba Selective** | `recurrent::mamba_selective` | `mamba_selective_scan_oracle` | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | —                                    |
| **Recurrent — RWKV**            | `recurrent::rwkv`            | `rwkv_oracle`                 | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | —                                    |
| **Diffusion — LLaDA**           | `diffusion::decoder`         | `llada_denoise_oracle`        | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | no LLaDA checkpoint in V1            |
| **Diffusion — Dream**           | `diffusion::decoder`         | `dream_remask_oracle`         | ⚪                           | �             | ⚪      | ⚪      | ⚪        | no Dream checkpoint in V1            |
| **Quantized — Q4 GEMM**         | `quantized::q4_gemm`         | `q4_gemm_oracle`              | 🟡 (Qwen3.5-0.8B-OptiQ-4bit) | ⚪            | 🟢      | ⚪      | ⚪        | —                                    |
| **Speculative — Draft/Verify**  | `speculative::*`             | `speculative_oracle`          | ⚪                           | ⚪            | ⚪      | ⚪      | ⚪        | out of V1 scope                      |

## Coverage summary

| Signal                         | Families with 🟢 | Families with 🟡 |
| ------------------------------ | ---------------- | ---------------- |
| Real Model                     | 0                | 2 (GQA, Q4 GEMM) |
| Agentic Trace                  | 0                | 0                |
| NIAH (8k)                      | 2                | 0                |
| NIAH (32k)                     | 0                | 1 (GQA)          |
| Quality (MMLU/GPQA/perplexity) | 0                | 0                |
| Stability (50 runs)            | 0                | 0                |

## Promotion gates (V1 → V2 → G1)

- **V0 → V1** (current target): at least one family at 🟢 in each
  signal column above.
- **V1 → V2**: all families at 🟢 or 🟡 with no � blockers; one family
  per category (attention, MoE, recurrent, quantized) at full 🟢.
- **V2 → G1** (general availability): all 22 family rows 🟢.

## How to update this matrix

Each row maps 1:1 to a `*_oracle` function in
`perf-core/model-kernels/src/<family>/`. When a CI signal flips from
🔴/⚪ to 🟢 or 🟡, update the row in a follow-up commit referencing
the run URL.

Cross-references:

- `perf-core/model-kernels/src/lib.rs` — facade layout.
- `perf-core/eval-harness/src/runner.rs` — harness that runs NIAH + MMLU.
- `portage/.github/workflows/harbor-gate.yml` — live NIAH oracle gate.
- `_cockpit/audit-phenotype-omlx.json` (BACKLOG-OMLX-003) — backlog tracking.
