//! Per-model-family end-to-end trace baselines.
//!
//! Each test pins down a different acceptance-matrix row from
//! `02_SPECIFICATIONS.md`:
//!
//! | Test | Family / row | Trace shape |
//! |---|---|---|
//! | `cca_block_baseline_round_trip` | ZAYA — CCA & compact nonlinear expert path | Three block summaries + query → softmax-weighted output |
//! | `mla_cache_baseline_round_trip` | DeepSeek — MLA, routed experts, proposal/MTP | Four `compressed_kv` + `k_rope` cache entries + query |
//! | `qwen_deltanet_moe_end_to_end_baseline_round_trip` | Qwen agentic — long-context decode, GQA or DeltaNet state, sparse MoE | DeltaNet recurrence over a four-token chunk → top-k sparse-MoE |
//! | `qwen_moe_end_to_end_v2_baseline_round_trip` | Qwen agentic — sparse-MoE per-stage composition (tiled GEMM + tiled reduce + writeback) | Router → dispatch → grouped_gemm_tiled → weighted_reduce_tiled → stage_expert_outputs + coalesced_writeback over `num_tokens=4, num_experts=3, top_k=2, hidden=4, k=4, capacity_factor=2.0` |
//! | `moe_writeback_2x8_baseline_round_trip` | Qwen agentic — turn-12 dispatch-aware DRAM writeback | top_k=2, 8 tokens, 3 experts, hidden=4, LCG-seeded expert outputs |
//! | `olmoe_moe_end_to_end_baseline_round_trip` | OLMoE-1B-7B — turn-14 generalized per-stage composition (64 experts, top_k=8, 1 shared expert) | Same per-stage chain as Qwen-MoE v2 with `num_tokens=4, num_experts=64, top_k=8, hidden=4, k=4, capacity_factor=1.5` |
//!
//! The per-family `seed` is part of the inputs so the input hash
//! distinguishes each trace from any other shape that happens to share
//! the same field names.
//!
//! Each family lives in its own sub-module. The `compute_*_output`
//! helpers are colocated with the family whose round-trip test owns
//! them (no helper is shared across families), which keeps the imports
//! in each sub-module tight and clippy-clean.

mod cca;
mod mla;
mod olmoe;
mod qwen;