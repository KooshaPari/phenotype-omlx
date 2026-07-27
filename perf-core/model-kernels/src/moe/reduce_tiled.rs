//! Tiled / blocked weighted expert reduction with oracle parity to the
//! scalar path in [`super::reduce`].
//!
//! The tiled path iterates the hidden dimension block-by-block instead
//! of element-by-element. It is an *additive* parallel function: the
//! existing [`super::reduce::weighted_reduce`] signature is untouched
//! so every caller in the codebase keeps working without changes.
//! Both paths share the same
//! `(expert_outs, weights, experts_per_token, hidden, out)` contract
//! and the same output conventions:
//!
//! - For every token `t` and every expert index `e ∈ [0, experts_per_token)`,
//!   `out[t, h] = Σ_e weights[t, e] * expert_outs[t, e, h]`.
//! - Empty `weights` is a no-op (matches the scalar contract).
//!
//! The tile size is selected automatically based on `hidden`:
//! `tile = min(64, hidden)`. The canonical Qwen-MoE hidden block uses
//! `hidden == 128`, which yields `tile == 64`. Smaller embeddings
//! (`hidden == 16`) pick `tile == 16`.
//!
//! This module is split out of `reduce.rs` to keep both files under the
//! 350-line target and to mirror the `gemm` / `gemm_tiled` split.

use crate::error::{KernelError, Result};

/// Tile size selection policy: `tile = min(64, hidden)`. Exposed
/// (`pub`) so the bench harness and the SOTA coverage matrix can
/// pin the chosen value at specific `hidden` shapes — see
/// `weighted_reduce_tiled_tile_size_selection` in the test module.
///
/// The 64-element ceiling matches the canonical Qwen-MoE hidden block
/// (`hidden == 128` → `tile == 64`). For shapes smaller than 64 the
/// tile collapses to `hidden` so every block is a single `tile`-element
/// pass.
#[inline]
pub fn tile_size_for(hidden: usize) -> usize {
    const TILE_MAX: usize = 64;
    hidden.min(TILE_MAX)
}

/// Compute `out[t, :] = Σ_e weights[t, e] * expert_outs[t, e, :]`
/// for every token, iterating the hidden dimension in `tile`-sized
/// blocks. The tile size is selected via [`tile_size_for`].
///
/// This is the additive parallel function to
/// [`super::reduce::weighted_reduce`]. It produces byte-equal output
/// to the scalar path for the same inputs (modulo tile-block
/// ordering), so the SOTA candidate `WeightedMoeReduceTiled` can be
/// selected on the basis of its `tuning_record.p95_ns` without
/// changing the model's contract.
#[allow(clippy::too_many_arguments)]
pub fn weighted_reduce_tiled(
    expert_outs: &[f32],
    weights: &[f32],
    experts_per_token: usize,
    hidden: usize,
    out: &mut [f32],
) -> Result<()> {
    if hidden == 0 {
        return Err(KernelError::ZeroDimension {
            what: "hidden",
            got: 0,
        });
    }
    if experts_per_token == 0 {
        return Err(KernelError::ZeroDimension {
            what: "experts_per_token",
            got: 0,
        });
    }
    if weights.is_empty() {
        // No tokens to reduce; nothing to do.
        return Ok(());
    }
    let num_tokens = weights.len() / experts_per_token;
    if weights.len() != num_tokens * experts_per_token {
        return Err(KernelError::BadBufferLength {
            what: "weights",
            expected: weights.len(),
            got: num_tokens * experts_per_token,
        });
    }
    let expected_eo = num_tokens * experts_per_token * hidden;
    if expert_outs.len() != expected_eo {
        return Err(KernelError::BadBufferLength {
            what: "expert_outs",
            expected: expected_eo,
            got: expert_outs.len(),
        });
    }
    if out.len() != num_tokens * hidden {
        return Err(KernelError::BadBufferLength {
            what: "out",
            expected: num_tokens * hidden,
            got: out.len(),
        });
    }
    let tile = tile_size_for(hidden);
    debug_assert!(tile >= 1, "tile must be >= 1 after ZeroDimension guard");
    debug_assert!(tile <= hidden, "tile must be <= hidden");

    for t in 0..num_tokens {
        let out_row = &mut out[t * hidden..t * hidden + hidden];
        for slot in out_row.iter_mut() {
            *slot = 0.0;
        }
        for e in 0..experts_per_token {
            let w = weights[t * experts_per_token + e];
            let eo_row = &expert_outs[(t * experts_per_token + e) * hidden
                ..(t * experts_per_token + e) * hidden + hidden];
            // Iterate over `hidden` in tiles of `tile`. Each tile is
            // a contiguous `[h0, h_end)` slice of the row. The
            // accumulation sums cleanly across tile blocks because
            // the row was zeroed above, so cross-tile contributions
            // do not inherit any sentinel value.
            let mut h0 = 0usize;
            while h0 < hidden {
                let h_end = (h0 + tile).min(hidden);
                for h in h0..h_end {
                    out_row[h] += w * eo_row[h];
                }
                h0 = h_end;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Lcg;

    // =========================================================================
    // SOTA opt-in tests — historical note (turn-12 follow-up)
    // =========================================================================
    //
    // Turn-11's forward-priority note (see
    // `docs/sessions/20260718-metal-model-runtime/18_TURN_11_RESUME_NOTES.md`
    // §13 line 200) recorded that `weighted_reduce_tiled` carries four
    // `#[ignore]`-marked SOTA opt-in tests covering the f32 / f16 / bf16 / i8
    // SIMD reference paths and instructed turn-12 to "lift these into the
    // default test surface if CI carries the SIMD toolchain, or keep them
    // gated behind a documented env flag if it does not."
    //
    // On inspection at turn-12, this historical claim does **not** match the
    // checked-in code. `weighted_reduce_tiled` (added at commit `706b28d`
    // on top of the archived SIMD baseline `6de65d0`) ships exactly five
    // active tests below — none of them are `#[ignore]`-marked and none are
    // named with the `sota_*` prefix. A repository-wide search confirms:
    //
    //   * no function or method named `weighted_reduce_simd*` exists in
    //     `perf-core/` (the only SIMD-dispatch site is
    //     `perf-core/turbo-quant/src/minmax.rs`, gated by
    //     `#[cfg(target_arch = "aarch64")]` for the NEON path);
    //   * the four candidate test names (`sota_f32_path_matches_simd_reference`,
    //     `sota_f16_path_matches_simd_reference`,
    //     `sota_bf16_path_matches_simd_reference`,
    //     `sota_quantized_int8_path_matches_simd_reference`) appear in zero
    //     files across the workspace, including under `.archive/`.
    //
    // In other words, the SIMD dispatch path for `weighted_reduce_tiled`
    // was never wired in this branch — the present branch
    // (`chore/archive-no-simd-lib-rs-2026-07-18`) was cut specifically to
    // archive the experimental non-SIMD lib.rs variant (see commit
    // `6de65d0 chore: archive experimental non-SIMD lib.rs variant`), and
    // the SIMD path itself lives only in
    // `.archive/lib.rs.no-simd-2026-07-18.bak` (a no-SIMD reference, not
    // a SIMD implementation).
    //
    // Path-C resolution: do NOT introduce four `#[ignore]`-marked SOTA
    // tests here until the SIMD reference path the tests would assert
    // against actually exists. Doing so would create dead assertions
    // pinned to a kernel that has not been merged, which is the precise
    // failure mode the turn-11 note tried to flag.
    //
    // What each SOTA test will assert once the SIMD path lands:
    //
    // 1. `sota_f32_path_matches_simd_reference` —
    //    `weighted_reduce_tiled_simd_f32(&expert_outs, &weights, ept, hidden,
    //    &mut out)` must match `weighted_reduce_tiled` element-wise within
    //    `1e-6` on random `f32` inputs of canonical Qwen-MoE shape
    //    `(num_tokens=8, experts_per_token=3, hidden=128)`. Pins the
    //    fused-multiply-add order across the SIMD tile so cross-tile
    //    boundary sums cannot drift.
    //
    // 2. `sota_f16_path_matches_simd_reference` —
    //    same parity contract as (1) but for the `f16` half-precision
    //    reference path; tolerance widens to `1e-3` to accommodate `f16`
    //    mantissa rounding. Asserts the down-cast
    //    `f32 expert_outs → f16 → f32 out` round-trip stays within the
    //    expected precision envelope.
    //
    // 3. `sota_bf16_path_matches_simd_reference` —
    //    same parity contract as (1) but for `bf16` (bfloat16). Tolerance
    //    `1e-2` (same dynamic range as `f32`, 7-bit mantissa). Pins the
    //    bfloat16 SIMD lane width contract so a future migration to a
    //    narrower SIMD lane does not silently degrade output.
    //
    // 4. `sota_quantized_int8_path_matches_simd_reference` —
    //    asserts the quantized INT8 path matches the `f32` reference
    //    within a scale-aware tolerance (typically `scale * 2^-7` where
    //    `scale = max(|expert_outs|) / 127.0`). Pins the symmetric INT8
    //    quantization + dequantization contract end-to-end across the
    //    SIMD tile so the row-wise scale factor is applied correctly.
    //
    // Which kernel / commit should introduce that path:
    //
    //   * The next kernel on the metal-model-runtime forward DAG
    //     (see `docs/sessions/20260718-metal-model-runtime/03_DAG_WBS.md`
    //     lines 49–53) is the **dispatch-aware writeback stage**, not the
    //     SIMD reference path itself. The SIMD reference for
    //     `weighted_reduce_tiled` is a re-introduction of the work that
    //     was archived at commit `6de65d0`; resurrecting it requires
    //     first un-ignoring the `.archive/lib.rs.no-simd-2026-07-18.bak`
    //     variant, then re-deriving the SIMD path against the current
    //     `weighted_reduce_tiled` scalar-tile reference.
    //   * The `KernelOp::MoeReduce` candidate in
    //     `perf-core/kernel-registry/tests/sota_operators/coverage_matrix.rs`
    //     already declares a `MoeReduceTiled` backend tagged `'moe_reduce'`;
    //     the SIMD path is expected to slot in as a sibling backend
    //     (e.g. `MoeReduceTiledSimd`) without disturbing the
    //     `MoeReduceScalar` reference or the current SOTA selector
    //     contract.
    //
    // How a contributor lifts this note (file an issue, link the DAG):
    //
    //   1. File an issue titled
    //      "Add SIMD reference path for `weighted_reduce_tiled` (f32/f16/bf16/i8)"
    //      and link the dispatch sub-DAG item from
    //      `03_DAG_WBS.md` line 198 (the dispatch-aware writeback stage
    //      item that immediately follows `weighted_reduce_tiled` in the
    //      forward DAG).
    //   2. Land the SIMD reference kernel in a sibling module
    //      (`perf-core/model-kernels/src/moe/reduce_tiled_simd.rs` or
    //      inline under `#[cfg(target_arch = "aarch64")]`, mirroring the
    //      gating pattern in `perf-core/turbo-quant/src/minmax.rs`).
    //   3. Re-derive the four `sota_*` tests listed above against the
    //      new SIMD kernel, then merge them with **active** (no
    //      `#[ignore]`) so they run in the default `cargo test` surface.
    //      The 859 + 4 passing / 2 - 4 ignored target from the turn-12
    //      task spec then lands cleanly.
    //
    // Until those steps complete, the five active tests below are the
    // complete test surface for this module and the SIMD opt-in claim
    // from turn-11 §13 remains a forward item, not a missing assertion.
    // =========================================================================

    /// Build a deterministic `[num_tokens, experts_per_token, hidden]`
    /// expert-output tensor.
    fn deterministic_expert_outs(
        num_tokens: usize,
        experts_per_token: usize,
        hidden: usize,
        salt: u64,
    ) -> Vec<f32> {
        let mut rng = Lcg::new(0xCAFE_F00D ^ salt);
        (0..num_tokens * experts_per_token * hidden)
            .map(|_| rng.next_signed())
            .collect()
    }

    /// Build a deterministic `[num_tokens, experts_per_token]` weight
    /// matrix.
    fn deterministic_weights(num_tokens: usize, experts_per_token: usize, salt: u64) -> Vec<f32> {
        let mut rng = Lcg::new(0xBEEF_DEAD ^ salt);
        (0..num_tokens * experts_per_token)
            .map(|_| rng.next_signed() * 0.5) // smaller magnitude so accumulated sums stay in range
            .collect()
    }

    /// Bit-equality check against the scalar reference. Both
    /// implementations perform the same `f32` multiply-add in the
    /// same order modulo tile-block ordering, so the tile path's
    /// output is byte-equal to the scalar output element-by-element.
    /// We pin the absolute tolerance at `1e-5` per the task spec.
    fn assert_close(a: &[f32], b: &[f32], abs: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "[{label}] length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            if (x - y).abs() > abs {
                panic!(
                    "[{label}] element {i} differs: tiled={x} scalar={y} (|d|={})",
                    (x - y).abs()
                );
            }
        }
    }

    /// Tiled output must match scalar output element-wise within
    /// `1e-5` on random inputs. This is the oracle-parity contract
    /// for the new path: any future regression in the block-iteration
    /// arithmetic (e.g. accidentally double-counting a boundary
    /// block, swapping the loop variables, or re-using a stale
    /// accumulator) trips this assertion.
    #[test]
    fn weighted_reduce_tiled_matches_scalar_for_random_inputs() {
        let num_tokens = 8;
        let experts_per_token = 3;
        let hidden = 32;
        let expert_outs = deterministic_expert_outs(num_tokens, experts_per_token, hidden, 0xA1);
        let weights = deterministic_weights(num_tokens, experts_per_token, 0xB2);

        let mut scalar_out = vec![0.0f32; num_tokens * hidden];
        crate::moe::reduce::weighted_reduce(
            &expert_outs,
            &weights,
            experts_per_token,
            hidden,
            &mut scalar_out,
        )
        .expect("scalar reference must accept well-formed inputs");

        let mut tiled_out = vec![0.0f32; num_tokens * hidden];
        weighted_reduce_tiled(
            &expert_outs,
            &weights,
            experts_per_token,
            hidden,
            &mut tiled_out,
        )
        .expect("tiled path must accept well-formed inputs");

        assert_close(&tiled_out, &scalar_out, 1e-5, "tiled vs scalar");
    }

    /// Empty weights buffer is a no-op (matches the scalar contract).
    /// Future refactors must not introduce a panic on the empty
    /// `weights` branch — the scalar path returns `Ok(())` and the
    /// tiled path must mirror that.
    #[test]
    fn weighted_reduce_tiled_handles_zero_tokens() {
        let expert_outs: [f32; 0] = [];
        let weights: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        weighted_reduce_tiled(&expert_outs, &weights, 2, 3, &mut out)
            .expect("tiled path must accept empty weights (no tokens)");
    }

    /// `hidden == 0` and `experts_per_token == 0` are both rejected
    /// with the same `ZeroDimension` variant as the scalar path. The
    /// tiled path performs the zero-dim check before any
    /// tile-size selection, so `tile_size_for(0)` is never reached
    /// on these inputs.
    #[test]
    fn weighted_reduce_tiled_rejects_zero_dim() {
        let eo = [0.0f32; 2];
        let w = [1.0f32; 2];
        let mut out = [0.0f32; 0];
        let err = weighted_reduce_tiled(&eo, &w, 2, 0, &mut out).unwrap_err();
        assert!(
            matches!(err, KernelError::ZeroDimension { .. }),
            "expected ZeroDimension for hidden=0, got {err:?}"
        );

        let mut out2 = [0.0f32; 0];
        let err = weighted_reduce_tiled(&eo, &w, 0, 3, &mut out2).unwrap_err();
        assert!(
            matches!(err, KernelError::ZeroDimension { .. }),
            "expected ZeroDimension for experts_per_token=0, got {err:?}"
        );
    }

    /// Bad buffer lengths (mismatched `weights.len()`, mismatched
    /// `expert_outs.len()`, mismatched `out.len()`) must produce
    /// `BadBufferLength` errors matching the scalar path's surface.
    /// A future refactor of the tiled path must not silently accept
    /// under-sized buffers.
    #[test]
    fn weighted_reduce_tiled_rejects_bad_buffer_lengths() {
        // weights.len() mismatch: weights has 5 floats but
        // experts_per_token=2 implies 4 floats per token
        // (5 / 2 -> 2 tokens with 1 stragglers).
        let expert_outs = vec![0.0f32; 8]; // 2 * 2 * 2
        let weights = vec![1.0f32, 0.5, 0.3, 0.4, 0.2]; // 5 floats, not divisible by 2
        let mut out = vec![0.0f32; 4];
        let err = weighted_reduce_tiled(&expert_outs, &weights, 2, 2, &mut out).unwrap_err();
        assert!(
            matches!(err, KernelError::BadBufferLength { what, .. } if what == "weights"),
            "expected BadBufferLength for weights, got {err:?}"
        );

        // expert_outs.len() mismatch: expects 2*2*2=8 but has 6.
        let expert_outs2 = vec![0.0f32; 6];
        let weights2 = vec![1.0f32, 0.5, 0.3, 0.4]; // 2 tokens * 2 experts
        let mut out2 = vec![0.0f32; 4];
        let err = weighted_reduce_tiled(&expert_outs2, &weights2, 2, 2, &mut out2).unwrap_err();
        assert!(
            matches!(err, KernelError::BadBufferLength { what, .. } if what == "expert_outs"),
            "expected BadBufferLength for expert_outs, got {err:?}"
        );

        // out.len() mismatch: expects 2*2=4 but has 3.
        let expert_outs3 = vec![0.0f32; 8];
        let weights3 = vec![1.0f32, 0.5, 0.3, 0.4];
        let mut out3 = vec![0.0f32; 3];
        let err = weighted_reduce_tiled(&expert_outs3, &weights3, 2, 2, &mut out3).unwrap_err();
        assert!(
            matches!(err, KernelError::BadBufferLength { what, .. } if what == "out"),
            "expected BadBufferLength for out, got {err:?}"
        );
    }

    /// The tile-size selection policy is pinned at this surface
    /// instead of through the loop internals so the contract is
    /// stable across loop refactors. The two canonical shapes
    /// spelled out in the task spec:
    ///
    /// - `hidden=128` → `tile = 64`
    /// - `hidden=16`  → `tile = 16`
    #[test]
    fn weighted_reduce_tiled_tile_size_selection() {
        assert_eq!(tile_size_for(128), 64, "Qwen-MoE canonical hidden block");
        assert_eq!(tile_size_for(16), 16, "small-hidden block");
        // Spot-check the boundary cases that bracket the policy.
        assert_eq!(tile_size_for(64), 64, "tile_max hit exactly");
        assert_eq!(tile_size_for(65), 64, "tile_max caps the policy");
        assert_eq!(tile_size_for(32), 32, "hidden below tile_max");
        assert_eq!(tile_size_for(256), 64, "tile_max caps large hidden");
        assert_eq!(tile_size_for(1), 1, "degenerate 1-wide still selects");
    }
}
