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
        return Err(KernelError::ZeroDimension { what: "hidden", got: 0 });
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
    fn deterministic_weights(
        num_tokens: usize,
        experts_per_token: usize,
        salt: u64,
    ) -> Vec<f32> {
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
        let expert_outs =
            deterministic_expert_outs(num_tokens, experts_per_token, hidden, 0xA1);
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
        let err =
            weighted_reduce_tiled(&expert_outs, &weights, 2, 2, &mut out).unwrap_err();
        assert!(
            matches!(err, KernelError::BadBufferLength { what, .. } if what == "weights"),
            "expected BadBufferLength for weights, got {err:?}"
        );

        // expert_outs.len() mismatch: expects 2*2*2=8 but has 6.
        let expert_outs2 = vec![0.0f32; 6];
        let weights2 = vec![1.0f32, 0.5, 0.3, 0.4]; // 2 tokens * 2 experts
        let mut out2 = vec![0.0f32; 4];
        let err =
            weighted_reduce_tiled(&expert_outs2, &weights2, 2, 2, &mut out2).unwrap_err();
        assert!(
            matches!(err, KernelError::BadBufferLength { what, .. } if what == "expert_outs"),
            "expected BadBufferLength for expert_outs, got {err:?}"
        );

        // out.len() mismatch: expects 2*2=4 but has 3.
        let expert_outs3 = vec![0.0f32; 8];
        let weights3 = vec![1.0f32, 0.5, 0.3, 0.4];
        let mut out3 = vec![0.0f32; 3];
        let err =
            weighted_reduce_tiled(&expert_outs3, &weights3, 2, 2, &mut out3).unwrap_err();
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