//! Mixture-of-Depths (MoD) sparse token routing.
//!
//! MoD (Raposo et al., 2024) is a sparse routing scheme that decides, on
//! every transformer layer, **which tokens** are allowed to enter the
//! block's heavy compute path. Unlike MoE — which selects *which expert*
//! processes each token — MoD selects *which tokens* are processed at all;
//! the surviving tokens share a single expert (or a small fixed stack of
//! experts) while the rest are carried around unchanged.
//!
//! This module is the scalar oracle for the family. It is intentionally
//! minimal:
//!
//! - [`mod_route`] reads per-token router weights, applies a sigmoid
//!   score, and picks the top-`k` survivors by score with ties broken by
//!   *lower index* (the determinism contract used by the kernel
//!   registry's selector).
//! - [`mod_apply`] materializes a contiguous `[k, dim]` buffer of the
//!   surviving rows so downstream kernels can run dense compute.
//! - [`mod_scatter_back`] is the inverse: it scatters the processed rows
//!   back into a `[num_tokens, dim]` buffer and fills the skipped
//!   positions with a caller-supplied constant (typically `0.0`).
//!
//! All functions are pure: no RNG, no FFI, no `unsafe`. The reference
//! scoring function is deterministic so test oracles can be reproduced
//! byte-for-byte. The capacity-factor argument lives in `(0, 1]`;
//! `1.0` means "route every token" (identity) and `0 < c < 1` selects
//! `floor(c * num_tokens)` survivors.
//!
//! # Layout
//!
//! - `weights: [num_tokens]` per-token router weights (typically a small
//!   linear projection of the hidden state followed by a sigmoid).
//! - `full_hidden_states: [num_tokens * dim]` row-major.
//! - `ModRoutePlan::selected_tokens`: survivor indices in input order.
//! - `ModRoutePlan::capacity_factor`: the capacity the plan was built
//!   under; downstream kernels may scale residual streams by
//!   `1 / capacity_factor` to preserve the unconditional expectation
//!   of the per-token contribution.

use crate::error::{KernelError, Result};

/// Plan describing which tokens survive the current MoD layer.
///
/// `selected_tokens` is sorted by descending score (ties broken by lower
/// index). `capacity_factor` is the capacity the plan was constructed
/// under, copied from [`ModRouterConfig::capacity_factor`].
#[derive(Debug, Clone, PartialEq)]
pub struct ModRoutePlan {
    /// Indices of the tokens that survive this MoD layer. Sorted by
    /// score descending, ties broken by lower index. Empty when
    /// `num_tokens == 0` or when the caller asks for zero survivors.
    pub selected_tokens: Vec<u32>,
    /// The capacity factor used to build the plan. Always in `(0, 1]`.
    pub capacity_factor: f32,
}

/// Configuration for a single MoD routing decision.
///
/// `capacity_factor` lives in `(0, 1]`. `1.0` selects every token
/// (identity routing); `mean_capacity` is the *target* mean number of
/// tokens surviving per layer and is informational — the kernel uses
/// `capacity_factor` to size the survivor count deterministically.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModRouterConfig {
    /// Fraction of tokens that survive per layer. Must satisfy
    /// `0 < capacity_factor <= 1` and be finite.
    pub capacity_factor: f32,
    /// Mean target capacity (tokens per layer) reported by the model
    /// config. Carried alongside `capacity_factor` so callers can
    /// reconcile the fraction with the absolute count.
    pub mean_capacity: f32,
}

/// Compute the survivor plan for a layer.
///
/// Algorithm:
/// 1. Validate `capacity_factor ∈ (0, 1]`.
/// 2. For each input weight, compute `score = sigmoid(weight)`.
/// 3. Sort by `(score desc, index asc)`.
/// 4. Take the top `k = floor(capacity_factor * num_tokens)` survivors.
/// 5. Return the survivor indices in score-desc order.
///
/// The scoring function is deterministic and never consults an RNG;
/// every call on the same inputs returns an equal plan.
pub fn mod_route(weights: &[f32], cfg: &ModRouterConfig) -> Result<ModRoutePlan> {
    if !cfg.capacity_factor.is_finite() || cfg.capacity_factor <= 0.0 || cfg.capacity_factor > 1.0 {
        return Err(KernelError::OutOfRange {
            what: "capacity_factor",
            min: 0.0,
            max: 1.0,
            got: cfg.capacity_factor,
        });
    }
    if !cfg.mean_capacity.is_finite() || cfg.mean_capacity < 0.0 {
        return Err(KernelError::OutOfRange {
            what: "mean_capacity",
            min: 0.0,
            max: f32::INFINITY,
            got: cfg.mean_capacity,
        });
    }
    let n = weights.len();
    if n == 0 {
        return Ok(ModRoutePlan {
            selected_tokens: Vec::new(),
            capacity_factor: cfg.capacity_factor,
        });
    }
    // floor(capacity_factor * n). Saturating cast: capacity_factor is in
    // (0, 1] and n is non-negative, so the product is non-negative and
    // finite. Cast to usize via floor.
    let cap = cfg.capacity_factor * n as f32;
    let k = cap.floor() as usize;
    // Edge case: capacity_factor > 0 but `k == 0`. Promote to at least
    // one survivor so the downstream pipeline always has something to
    // run on. This matches the "always at least one token" convention
    // used by MoD papers in their smallest-cap configurations.
    let k = k.max(1).min(n);

    // Score and pair with the original index.
    let mut indexed: Vec<(usize, f32)> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| (i, sigmoid(w)))
        .collect();
    // Sort by (score desc, index asc).
    indexed.sort_by(|(ia, sa), (ib, sb)| {
        sb.partial_cmp(sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ia.cmp(ib))
    });

    let selected_tokens: Vec<u32> = indexed
        .iter()
        .take(k)
        .map(|(i, _)| *i as u32)
        .collect();
    Ok(ModRoutePlan {
        selected_tokens,
        capacity_factor: cfg.capacity_factor,
    })
}

/// Materialize the surviving rows of `full_hidden_states` as a
/// contiguous `[k, dim]` buffer.
///
/// `full_hidden_states` is laid out `[num_tokens, dim]` row-major.
/// `dim` must be strictly positive; `full_hidden_states.len()` must
/// equal `plan.selected_tokens.len().max(num_tokens) * dim` modulo
/// rows that have not been read — we validate by ensuring the plan's
/// every selected index is in-range and that the buffer length is
/// consistent with the largest referenced index.
pub fn mod_apply(
    plan: &ModRoutePlan,
    full_hidden_states: &[f32],
    dim: usize,
) -> Result<Vec<f32>> {
    if dim == 0 {
        return Err(KernelError::ZeroDimension { what: "dim", got: 0 });
    }
    let k = plan.selected_tokens.len();
    let mut out = Vec::with_capacity(k.saturating_mul(dim));
    for &idx in &plan.selected_tokens {
        let row = idx as usize;
        let start = row.checked_mul(dim).ok_or(KernelError::BadBufferLength {
            what: "selected_token row index * dim",
            expected: 0,
            got: row * dim,
        })?;
        let end = start + dim;
        if end > full_hidden_states.len() {
            return Err(KernelError::BadBufferLength {
                what: "full_hidden_states",
                expected: end,
                got: full_hidden_states.len(),
            });
        }
        out.extend_from_slice(&full_hidden_states[start..end]);
    }
    Ok(out)
}

/// Inverse of [`mod_apply`]: scatter the processed rows back into a
/// full-size buffer, filling every position that was skipped with
/// `fill`.
///
/// Returns the full-size buffer; the caller does not need to allocate
/// it. Skipped positions are filled with `fill` (typically `0.0` so
/// the residual stream carries the previous value untouched, or a
/// residual scaling factor when the model uses weighted carries).
///
/// `full_len` is the *number of tokens* in the full buffer (so the
/// returned buffer has `full_len * dim` elements).
pub fn mod_scatter_back(
    selected: &[f32],
    plan: &ModRoutePlan,
    full_len: usize,
    dim: usize,
    fill: f32,
) -> Result<Vec<f32>> {
    if dim == 0 {
        return Err(KernelError::ZeroDimension { what: "dim", got: 0 });
    }
    if selected.len() != plan.selected_tokens.len() * dim {
        return Err(KernelError::BadBufferLength {
            what: "selected",
            expected: plan.selected_tokens.len() * dim,
            got: selected.len(),
        });
    }
    let mut out = vec![fill; full_len * dim];
    for (slot, &idx) in plan.selected_tokens.iter().enumerate() {
        let row = idx as usize;
        if row >= full_len {
            return Err(KernelError::BadBufferLength {
                what: "selected_token row index",
                expected: full_len,
                got: row,
            });
        }
        let src_start = slot * dim;
        let dst_start = row * dim;
        out[dst_start..dst_start + dim]
            .copy_from_slice(&selected[src_start..src_start + dim]);
    }
    Ok(out)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(capacity_factor: f32) -> ModRouterConfig {
        ModRouterConfig {
            capacity_factor,
            mean_capacity: 0.0,
        }
    }

    #[test]
    fn capacity_one_returns_all_tokens_in_score_descending_order() {
        // Sigmoid is monotonic, so the score-desc order over all
        // tokens is the same as weight-desc order. With weights
        // [0.1, -0.2, 0.3, -0.4, 0.5] the top-scoring tokens are 4
        // (w=0.5), 2 (w=0.3), 0 (w=0.1), 1 (w=-0.2), 3 (w=-0.4).
        let weights = vec![0.1f32, -0.2, 0.3, -0.4, 0.5];
        let plan = mod_route(&weights, &cfg(1.0)).unwrap();
        assert_eq!(plan.selected_tokens, vec![4, 2, 0, 1, 3]);
    }

    #[test]
    fn capacity_half_returns_top_half_by_sigmoid_score() {
        // Weights are monotonically increasing -> sigmoid scores are
        // strictly increasing -> top half is the high-weight tail.
        let weights: Vec<f32> = (0..10).map(|i| (i as f32 - 4.5) * 2.0).collect();
        let plan = mod_route(&weights, &cfg(0.5)).unwrap();
        assert_eq!(plan.selected_tokens.len(), 5);
        // Strictly increasing scores => top 5 are the last 5 indices,
        // listed score-descending (which equals index-descending here).
        assert_eq!(plan.selected_tokens, vec![9, 8, 7, 6, 5]);
    }

    #[test]
    fn deterministic_across_runs() {
        let weights = vec![0.2f32, 0.9, -0.4, 0.7, 0.0, 0.5];
        let p1 = mod_route(&weights, &cfg(0.5)).unwrap();
        let p2 = mod_route(&weights, &cfg(0.5)).unwrap();
        assert_eq!(p1, p2);
    }

    #[test]
    fn empty_weights_yields_empty_plan() {
        let plan = mod_route(&[], &cfg(0.5)).unwrap();
        assert!(plan.selected_tokens.is_empty());
        assert_eq!(plan.capacity_factor, 0.5);
    }

    #[test]
    fn ties_break_by_lower_index() {
        // Identical weights -> identical sigmoid scores -> sort by
        // index ascending among ties.
        let weights = vec![0.5f32; 6];
        let plan = mod_route(&weights, &cfg(0.5)).unwrap();
        assert_eq!(plan.selected_tokens, vec![0, 1, 2]);
    }

    #[test]
    fn apply_scatter_round_trip_with_zero_fill_is_exact() {
        let weights = vec![0.1f32, 0.5, -0.2, 0.9, 0.0, 0.3];
        let plan = mod_route(&weights, &cfg(0.5)).unwrap();
        let num_tokens = weights.len();
        let dim = 3;
        // Build a deterministic full buffer: hidden[t, d] = t * 10 + d.
        let full: Vec<f32> = (0..num_tokens * dim)
            .map(|i| (i / dim) as f32 * 10.0 + (i % dim) as f32)
            .collect();
        // Apply: extract the surviving rows.
        let selected = mod_apply(&plan, &full, dim).unwrap();
        // Scatter back with fill=0.0; the result must match `full` at
        // surviving rows and be 0.0 at the rest.
        let mut scattered = mod_scatter_back(&selected, &plan, num_tokens, dim, 0.0).unwrap();
        // Zero out the non-survivor rows on a reference copy.
        let mut expected = full.clone();
        let survivor: std::collections::HashSet<u32> =
            plan.selected_tokens.iter().copied().collect();
        for t in 0..num_tokens {
            if !survivor.contains(&(t as u32)) {
                for d in 0..dim {
                    expected[t * dim + d] = 0.0;
                }
            }
        }
        // L2 sum of squared diffs must be 0.
        let l2: f64 = scattered
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| {
                let d = (*a as f64) - (*b as f64);
                d * d
            })
            .sum();
        assert_eq!(l2, 0.0, "round-trip should be exact with fill=0.0; got L2={l2}");
        // Sanity: the surviving rows really are the original rows.
        for &idx in &plan.selected_tokens {
            let row = idx as usize;
            for d in 0..dim {
                assert_eq!(scattered[row * dim + d], full[row * dim + d]);
            }
        }
        // Defensive: scattered must be the same length as the full
        // buffer.
        assert_eq!(scattered.len(), full.len());
        // Silence unused-mut on scattered (the binding is reassigned
        // by .unwrap() above but the variable would be flagged if we
        // didn't touch it again).
        let _ = &mut scattered;
    }

    #[test]
    fn rejects_capacity_factor_outside_unit_interval() {
        let weights = vec![0.1f32, 0.2, 0.3];
        // Zero: must reject.
        let err = mod_route(&weights, &cfg(0.0)).unwrap_err();
        assert!(matches!(err, KernelError::OutOfRange { .. }));
        // Negative: must reject.
        let err = mod_route(&weights, &cfg(-0.1)).unwrap_err();
        assert!(matches!(err, KernelError::OutOfRange { .. }));
        // Above 1: must reject.
        let err = mod_route(&weights, &cfg(1.5)).unwrap_err();
        assert!(matches!(err, KernelError::OutOfRange { .. }));
        // NaN: must reject.
        let err = mod_route(&weights, &cfg(f32::NAN)).unwrap_err();
        assert!(matches!(err, KernelError::OutOfRange { .. }));
    }

    #[test]
    fn capacity_factor_one_is_accepted_and_returns_all_tokens() {
        // Strictly increasing weights => score-desc order is the same
        // as index-desc order, so all tokens appear, in reverse index
        // order.
        let weights = vec![0.1f32, 0.2, 0.3, 0.4];
        let plan = mod_route(&weights, &cfg(1.0)).unwrap();
        assert_eq!(plan.selected_tokens, vec![3, 2, 1, 0]);
        assert_eq!(plan.capacity_factor, 1.0);
    }

    #[test]
    fn very_small_capacity_promotes_at_least_one_survivor() {
        // capacity_factor * num_tokens floors to 0; the kernel must
        // promote k to 1 so the layer still has work to do.
        let weights = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let plan = mod_route(&weights, &cfg(0.0001)).unwrap();
        assert_eq!(plan.selected_tokens.len(), 1);
    }

    #[test]
    fn scatter_rejects_mismatched_selected_length() {
        let plan = ModRoutePlan {
            selected_tokens: vec![0, 2],
            capacity_factor: 0.5,
        };
        let dim = 3;
        // Selected has the wrong length (would need 6 elements for 2 rows of 3).
        let bad = vec![0.0f32; 5];
        let err = mod_scatter_back(&bad, &plan, 4, dim, 0.0).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn apply_rejects_zero_dim() {
        let plan = ModRoutePlan {
            selected_tokens: vec![0],
            capacity_factor: 1.0,
        };
        let err = mod_apply(&plan, &[0.0, 0.0, 0.0], 0).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }
}