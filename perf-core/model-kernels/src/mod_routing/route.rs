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

    let selected_tokens: Vec<u32> = indexed.iter().take(k).map(|(i, _)| *i as u32).collect();
    Ok(ModRoutePlan {
        selected_tokens,
        capacity_factor: cfg.capacity_factor,
    })
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
