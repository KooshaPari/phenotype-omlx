//! Dispatch-aware DRAM writeback for MoE expert outputs.
//!
//! The next kernel in the Metal model-runtime MoE DAG after
//! [`super::reduce_tiled::weighted_reduce_tiled`]. The motivation is
//! the layout mismatch between the per-expert contiguous GEMM
//! outputs and the per-token row-major residual buffer that the
//! host-side model loader expects. The kernel is split into two
//! additive halves:
//!
//! 1. [`stage_expert_outputs`] packs the host's
//!    `[num_tokens, 1, hidden]` activation tensor into per-expert
//!    contiguous blocks `[num_experts, capacity_used[e], hidden]`.
//! 2. [`coalesced_writeback`] walks the staged blocks and populates
//!    `[num_tokens, hidden]` in token-major coalesced order.
//!
//! All algorithms are deterministic given the inputs. The public
//! surface is additive — `grouped_gemm_tiled` and
//! `weighted_reduce_tiled` signatures are untouched.

use crate::error::{KernelError, Result};
use crate::moe::dispatch::DispatchPlan;

/// Sentinel component for dropped tokens in [`WritebackPlan::token_to_expert_slot`].
pub(crate) const DROPPED: usize = usize::MAX;

/// Tile size selection: `tile = min(64, hidden)`. Must stay
/// byte-equal to [`super::reduce_tiled::tile_size_for`] — pinned
/// by the `tile_size_for_matches_reduce_tiled_policy` test.
#[inline]
pub fn tile_size_for(hidden: usize) -> usize {
    hidden.min(64)
}

/// Per-expert staging buffer + token-to-(expert, slot) reverse map.
#[derive(Debug, Clone, PartialEq)]
pub struct WritebackPlan {
    /// `per_expert_blocks[e]` is `[capacity_used[e] * hidden]` floats.
    pub per_expert_blocks: Vec<Vec<f32>>,
    /// `token_to_expert_slot[t] = (expert_id, slot_in_expert_block)`
    /// for routed tokens, `(usize::MAX, usize::MAX)` for dropped.
    pub token_to_expert_slot: Vec<(usize, usize)>,
}

impl WritebackPlan {
    /// Expected length of the `expert_outs` buffer for
    /// `(num_tokens, experts_per_token, hidden)`. Mirrors the
    /// pre-existing `expected_eo` computation in `reduce_tiled`.
    #[inline]
    pub fn expected_eo_len(
        num_tokens: usize,
        experts_per_token: usize,
        hidden: usize,
    ) -> usize {
        num_tokens * experts_per_token * hidden
    }
}

/// Stage the host's per-token activations into per-expert contiguous
/// blocks. `expert_outs` is `[num_tokens, 1, hidden]` (top_k=1 in
/// this kernel — `token_to_expert_slot[t]` is a single tuple per
/// token). For every routed token `t` the row
/// `expert_outs[t, 0, :]` is copied into
/// `per_expert_blocks[e][slot * hidden .. slot * hidden + hidden]`
/// where `slot = expert_buckets[e].iter().position(|&x| x == t)`.
pub fn stage_expert_outputs(
    expert_outs: &[f32],
    dispatch_plan: &DispatchPlan,
    hidden: usize,
) -> Result<WritebackPlan> {
    if hidden == 0 {
        return Err(KernelError::ZeroDimension { what: "hidden", got: 0 });
    }
    let total_routed: usize = dispatch_plan.capacity_used.iter().sum();
    let total_input: usize = total_routed + dispatch_plan.dropped.len();
    let expected_eo = WritebackPlan::expected_eo_len(total_input, 1, hidden);
    if expert_outs.len() != expected_eo {
        return Err(KernelError::BadBufferLength {
            what: "expert_outs",
            expected: expected_eo,
            got: expert_outs.len(),
        });
    }
    let mut per_expert_blocks: Vec<Vec<f32>> = dispatch_plan
        .capacity_used
        .iter()
        .map(|&cap| vec![0.0f32; cap * hidden])
        .collect();
    let mut token_to_expert_slot: Vec<(usize, usize)> =
        vec![(DROPPED, DROPPED); total_input];
    for (e, bucket) in dispatch_plan.expert_buckets.iter().enumerate() {
        for (slot, &tok) in bucket.iter().enumerate() {
            let src = &expert_outs[tok * hidden..tok * hidden + hidden];
            let dst = &mut per_expert_blocks[e][slot * hidden..slot * hidden + hidden];
            dst.copy_from_slice(src);
            if tok < total_input {
                token_to_expert_slot[tok] = (e, slot);
            }
        }
    }
    Ok(WritebackPlan {
        per_expert_blocks,
        token_to_expert_slot,
    })
}

/// Populate `out[token_id, :]` from a [`WritebackPlan`], iterating
/// `hidden` in `tile = tile_size_for(hidden)` blocks. The writeback
/// is **idempotent** — `out` is zeroed first, so calling twice with
/// the same dispatch plan produces the same result. Dropped tokens
/// yield `out[t, :] = 0`.
pub fn coalesced_writeback(
    stage: &WritebackPlan,
    num_tokens: usize,
    hidden: usize,
    out: &mut [f32],
) -> Result<()> {
    if hidden == 0 {
        return Err(KernelError::ZeroDimension { what: "hidden", got: 0 });
    }
    if out.len() != num_tokens * hidden {
        return Err(KernelError::BadBufferLength {
            what: "out",
            expected: num_tokens * hidden,
            got: out.len(),
        });
    }
    for v in out.iter_mut() {
        *v = 0.0;
    }
    if num_tokens == 0 {
        return Ok(());
    }
    if stage.token_to_expert_slot.len() != num_tokens {
        return Err(KernelError::BadBufferLength {
            what: "token_to_expert_slot",
            expected: num_tokens,
            got: stage.token_to_expert_slot.len(),
        });
    }
    for block in &stage.per_expert_blocks {
        if block.len() % hidden != 0 {
            return Err(KernelError::BadBufferLength {
                what: "per_expert_blocks",
                expected: (block.len() / hidden) * hidden,
                got: block.len(),
            });
        }
    }
    let tile = tile_size_for(hidden);
    debug_assert!(tile >= 1);
    debug_assert!(tile <= hidden);
    for (t, &(expert, slot)) in stage.token_to_expert_slot.iter().enumerate() {
        if expert == DROPPED {
            continue;
        }
        let src_row =
            &stage.per_expert_blocks[expert][slot * hidden..slot * hidden + hidden];
        let dst_row = &mut out[t * hidden..t * hidden + hidden];
        let mut h = 0;
        while h < hidden {
            let step = tile.min(hidden - h);
            for i in 0..step {
                dst_row[h + i] += src_row[h + i];
            }
            h += step;
        }
    }
    Ok(())
}

// =============================================================================
// Tests (TDD — written first, before impl).
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Lcg;
    use crate::error::KernelError;
    use crate::moe::dispatch::{moe_dispatch, DispatchPlan};

    /// Build a deterministic dispatch plan + matching `expert_outs`.
    fn make_plan_and_outs(
        num_tokens: usize,
        num_experts: usize,
        hidden: usize,
        capacity_factor: f32,
        seed_assignments: u64,
        seed_outs: u64,
    ) -> (DispatchPlan, Vec<f32>) {
        let token_indices: Vec<usize> = (0..num_tokens).collect();
        let offset = (Lcg::new(seed_assignments).next_u64() % num_experts as u64) as usize;
        let assignments: Vec<(usize, f32)> = (0..num_tokens)
            .map(|t| ((t + offset) % num_experts, 1.0))
            .collect();
        let plan = moe_dispatch(&token_indices, &assignments, num_experts, capacity_factor)
            .expect("dispatch must accept well-formed inputs");
        let expert_outs: Vec<f32> = (0..num_tokens * hidden)
            .map(|_| Lcg::new(seed_outs).next_signed())
            .collect();
        (plan, expert_outs)
    }
    /// 1. byte-equality against a naive per-token copy.
    #[test]
    fn coalesced_writeback_matches_naive_per_token_sum() {
        let (plan, expert_outs) =
            make_plan_and_outs(17, 5, 64, 2.0, 0xA1, 0xB2);
        let stage = stage_expert_outputs(&expert_outs, &plan, 64).unwrap();
        let mut naive = vec![0.0f32; 17 * 64];
        for t in 0..17 {
            naive[t * 64..t * 64 + 64]
                .copy_from_slice(&expert_outs[t * 64..t * 64 + 64]);
        }
        let mut out = vec![f32::NAN; 17 * 64];
        coalesced_writeback(&stage, 17, 64, &mut out).unwrap();
        for (i, (&a, &b)) in naive.iter().zip(out.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "byte-eq at {i}: {a} vs {b}");
        }
    }
    /// 2. Stage preserves the per-expert layout in dispatch order.
    #[test]
    fn stage_expert_outputs_preserves_expert_layout() {
        let (plan, expert_outs) =
            make_plan_and_outs(9, 3, 4, 2.0, 0xC0FFEE, 0xDEAD);
        let stage = stage_expert_outputs(&expert_outs, &plan, 4).unwrap();
        for (e, &cap) in plan.capacity_used.iter().enumerate() {
            assert_eq!(stage.per_expert_blocks[e].len(), cap * 4);
        }
        for e in 0..3 {
            for (s, &t) in plan.expert_buckets[e].iter().enumerate() {
                for i in 0..4 {
                    assert_eq!(
                        stage.per_expert_blocks[e][s * 4 + i].to_bits(),
                        expert_outs[t * 4 + i].to_bits(),
                        "expert {e} slot {s} token {t} dim {i}"
                    );
                }
            }
        }
        for t in 0..9 {
            let (e, s) = stage.token_to_expert_slot[t];
            if e == DROPPED {
                assert_eq!(s, DROPPED);
                assert!(plan.dropped.contains(&t));
            } else {
                assert_eq!(plan.expert_buckets[e][s], t);
            }
        }
    }
    /// 3. tile_size_for matches the reduce_tiled policy.
    #[test]
    fn tile_size_for_matches_reduce_tiled_policy() {
        use crate::moe::reduce_tiled::tile_size_for as reduce_tile_size_for;
        for &h in &[1usize, 16, 32, 64, 65, 128, 256] {
            assert_eq!(tile_size_for(h), reduce_tile_size_for(h));
        }
    }
    /// 4. capacity_used <= 1 across all experts.
    #[test]
    fn writeback_handles_capacity_one() {
        let (plan, expert_outs) =
            make_plan_and_outs(5, 5, 8, 1.0, 0x42, 0x99);
        let stage = stage_expert_outputs(&expert_outs, &plan, 8).unwrap();
        for &cap in plan.capacity_used.iter() {
            assert!(cap <= 1);
        }
        let mut out = vec![0.0f32; 5 * 8];
        coalesced_writeback(&stage, 5, 8, &mut out).unwrap();
        for t in 0..5 {
            for h in 0..8 {
                assert_eq!(
                    out[t * 8 + h].to_bits(),
                    expert_outs[t * 8 + h].to_bits()
                );
            }
        }
    }
    /// 5. Zero hidden is rejected.
    #[test]
    fn writeback_rejects_zero_hidden() {
        let (plan, expert_outs) = make_plan_and_outs(3, 2, 1, 1.0, 1, 2);
        let err = stage_expert_outputs(&expert_outs, &plan, 0).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
        let stage = stage_expert_outputs(&expert_outs, &plan, 1).unwrap();
        let mut out: [f32; 0] = [];
        let err = coalesced_writeback(&stage, 3, 0, &mut out).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }
    /// 6. Buffer length mismatches are rejected.
    #[test]
    fn writeback_rejects_out_length_mismatch() {
        let (plan, expert_outs) = make_plan_and_outs(4, 2, 3, 2.0, 7, 11);
        let stage = stage_expert_outputs(&expert_outs, &plan, 3).unwrap();
        let mut out_short = vec![0.0f32; 4 * 3 - 1];
        let err = coalesced_writeback(&stage, 4, 3, &mut out_short).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { what: "out", .. }));
        let mut out_long = vec![0.0f32; 4 * 3 + 1];
        let err = coalesced_writeback(&stage, 4, 3, &mut out_long).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { what: "out", .. }));
        let mut bad_stage = stage.clone();
        if let Some(b) = bad_stage.per_expert_blocks.get_mut(0) {
            if !b.is_empty() {
                b.push(0.0);
            }
        }
        let mut out = vec![0.0f32; 12];
        let err = coalesced_writeback(&bad_stage, 4, 3, &mut out).unwrap_err();
        assert!(
            matches!(err, KernelError::BadBufferLength { what: "per_expert_blocks", .. }),
            "got {err:?}"
        );
    }
    /// 7. Uneven bucket sizes (e.g. {2, 0, 3}).
    #[test]
    fn writeback_handles_uneven_buckets() {
        let token_indices: Vec<usize> = (0..5).collect();
        let assignments: Vec<(usize, f32)> =
            vec![(0, 1.0), (2, 1.0), (0, 1.0), (2, 1.0), (2, 1.0)];
        let plan = moe_dispatch(&token_indices, &assignments, 3, 5.0).unwrap();
        assert_eq!(plan.capacity_used, vec![2, 0, 3]);
        let expert_outs: Vec<f32> =
            (0..20).map(|_| Lcg::new(0xFEED_FACE).next_signed()).collect();
        let mut naive = [0.0f32; 20];
        for t in 0..5 {
            naive[t * 4..t * 4 + 4]
                .copy_from_slice(&expert_outs[t * 4..t * 4 + 4]);
        }
        let stage = stage_expert_outputs(&expert_outs, &plan, 4).unwrap();
        assert_eq!(stage.per_expert_blocks[0].len(), 8);
        assert_eq!(stage.per_expert_blocks[1].len(), 0);
        assert_eq!(stage.per_expert_blocks[2].len(), 12);
        assert_eq!(stage.token_to_expert_slot[0], (0, 0));
        assert_eq!(stage.token_to_expert_slot[1], (2, 0));
        assert_eq!(stage.token_to_expert_slot[2], (0, 1));
        assert_eq!(stage.token_to_expert_slot[3], (2, 1));
        assert_eq!(stage.token_to_expert_slot[4], (2, 2));
        let mut out = vec![f32::NAN; 20];
        coalesced_writeback(&stage, 5, 4, &mut out).unwrap();
        for (i, (&a, &b)) in naive.iter().zip(out.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "byte-eq at {i}: {a} vs {b}");
        }
    }
    /// `expected_eo_len` helper contract.
    #[test]
    fn expected_eo_len_matches_product() {
        assert_eq!(WritebackPlan::expected_eo_len(0, 1, 4), 0);
        assert_eq!(WritebackPlan::expected_eo_len(8, 1, 4), 32);
        assert_eq!(WritebackPlan::expected_eo_len(8, 2, 4), 64);
        assert_eq!(WritebackPlan::expected_eo_len(64, 1, 128), 64 * 128);
    }
}
