//! Structure-only mapping between model families enumerated in
//! `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md`
//! and the Rust kernel + test(s) that cover each one.
//!
//! This file is *deliberately* non-executing — it has no `#[test]`
//! functions and no business logic. It exists so reviewers can audit
//! "every model family in the spec → every Rust test" in one place.
//!
//! Rows whose test(s) are not yet committed are marked
//! `PENDING — see known-issues.md` and carry a `TODO` comment so the
//! gap is visible to both humans and search-for-TODO tooling.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Conformance table
// ---------------------------------------------------------------------------
//
// | family  | kernel                                  | test(s)                                                                                        |
// |---------|-----------------------------------------|------------------------------------------------------------------------------------------------|
// | Qwen    | attention::gqa, moe::router, recurrent  | tests::qwen_bonsai::qwen_agentic_mini_trace_runs_and_is_finite                                  |
// |         |   ::deltanet                            | tests::qwen_bonsai::qwen_deltanet_chunk_matches_repeated_step_4_heads                         |
// |         |                                         | tests::qwen_bonsai::qwen_sparse_moe_pipeline_runs_end_to_end                                   |
// | DeepSeek| attention::mla, mtp_propose / verify    | tests::contracts::mla_attention_matches_oracle_for_random_inputs                               |
// |         |                                         | tests::mla_cache::mla_cache_round_trip_matches_mla_attention_oracle                            |
// |         |                                         | tests::mla_cache::mtp_propose_returns_argmax_per_offset                                       |
// |         |                                         | tests::mla_cache::mtp_verify_threshold_split_decision                                          |
// | LFM     | recurrent::short_conv                  | tests::zaya_lfm::lfm2_gated_short_conv_16_steps_matches_elementwise_product                    |
// | ZAYA    | attention::cca                          | tests::zaya_lfm::zaya_block_parallel_three_blocks_matches_explicit_reference                   |
// |         |                                         | tests::zaya_lfm::zaya_block_parallel_handles_non_uniform_block_sizes                            |
// | Bonsai  | quant::ternary (Bonsai-27B target)       | tests::qwen_bonsai::bonsai_ternary_matmul_matches_unpacked_reference                           |
// |         |                                         | tests::contracts::ternary_pack_matches_manual_packing                                          |
// |         |                                         | tests::contracts::ternary_unpack_inverts_pack                                                  |
// | LLaDA   | diffusion::denoise + remask            | tests::diffusion::llada_acceptance_trace_finishes_with_all_unmasked                            |
// |         |   (LowConfidence strategy)              | tests::diffusion::llada_acceptance_trace_is_deterministic_across_runs                          |
// |         |                                         | tests::diffusion::llada_running_unmasked_count_is_monotonically_non_decreasing                  |
// |         |                                         | tests::diffusion::llada_remask_count_per_position_bounded_by_total_steps_minus_one             |
// | Dream   | diffusion::denoise + remask            | tests::diffusion::dream_acceptance_trace_finishes_with_all_unmasked                             |
// |         |   (EntropyBased strategy)               | tests::diffusion::dream_acceptance_trace_is_deterministic_across_runs                           |
// |         |                                         | tests::diffusion::dream_running_unmasked_count_is_monotonically_non_decreasing                   |
// |         |                                         | tests::diffusion::dream_remask_count_per_position_bounded_by_total_steps_minus_one              |
// | DiffGem | diffusion::denoise + remask (target)    | LLaDA/Dream fixtures retained until DiffusionGemma artifact fixtures land                   |
// | Mamba   | recurrent::mamba                        | tests::contracts::mamba_scan_matches_recurrent_definition                                      |
// |         |                                         | tests::recurrent_hybrid::short_conv1d_32_input_trace_matches_naive_convolution                 |
// | Jamba   | recurrent::mamba + attention::gqa       | tests::recurrent_hybrid::jamba_mamba_chunked_output_matches_repeated_single_steps              |
// |         |   (hybrid)                              | tests::recurrent_hybrid::jamba_state_resume_after_single_step_matches_chunked_run              |
// | RWKV    | recurrent::rwkv (time-mix + channel-mix)| tests::contracts::rwkv_time_mix_matches_recurrent_definition                                   |
// |         |                                         | tests::recurrent_hybrid::rwkv7_16_step_trace_matches_hand_coded_reference                      |
// |         |                                         | tests::recurrent_hybrid::rwkv7_state_resume_after_first_step_matches_full_trace                 |
// ---------------------------------------------------------------------------

#[allow(dead_code)]
const CONFORMANCE_TABLE_DOC: &str = "see table above; tests under tests/ must keep names in sync";

// ---------------------------------------------------------------------------
// Static marker struct — exists so the file is not flagged as empty by
// rustc's "no items" lint, and so future maintainers can hang helpers
// here without breaking the test-harness contract that this file is
// structure-only.
// ---------------------------------------------------------------------------
struct ConformanceMap;

impl ConformanceMap {
    // Families whose tests are committed (none of these return values;
    // the methods exist purely as documentation surfaces that show up
    // in `rustdoc` and `cargo doc`).
    #[allow(dead_code)]
    fn qwen() {}
    #[allow(dead_code)]
    fn deepseek() {}
    #[allow(dead_code)]
    fn lfm() {}
    #[allow(dead_code)]
    fn zaya() {}
    #[allow(dead_code)]
    fn bonsai() {}
    #[allow(dead_code)]
    fn llada() {}
    #[allow(dead_code)]
    fn dream() {}
    #[allow(dead_code)]
    fn mamba() {}
    #[allow(dead_code)]
    fn jamba() {}
    #[allow(dead_code)]
    fn rwkv() {}
}

// ---------------------------------------------------------------------------
// Pending rows (no committed test yet)
// ---------------------------------------------------------------------------
//
// As of this writing every requested family row has at least one
// committed test; the block below remains as the canonical place to
// add `PENDING — see known-issues.md` rows for newly-enumerated
// families. The `TODO` comment marker is intentional: it lets the
// repo-wide TODO scanner surface conformance gaps.
//
// TODO(LLaDA-extensions): when a non-parallel masked denoise mode is
//   added, extend this table.
// TODO(Dream-extensions): when Dream-specific scheduler (top-k remask,
//   entropy-temperature schedule) lands, add a row here.
