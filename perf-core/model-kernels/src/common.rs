//! Shared helpers used by every kernel family.
//!
//! - [`approx_eq`]: floating-point equality suitable for oracle vs. kernel
//!   parity tests. The default tolerance (`abs = 1e-5`, `rel = 1e-4`) is the
//!   contract documented in `07_IMPLEMENTATION_PLAN.md`.
//! - [`Lcg`]: a tiny seeded PRNG. We deliberately avoid `rand` so that
//!   determinism in this crate is enforced structurally: every kernel that
//!   needs randomness builds its own [`Lcg`] from a `u64` seed.
//! - [`softmax_row`]: in-place softmax over a slice (shared by attention,
//!   diffusion, MoE router, etc.).

#![allow(dead_code)] // not every consumer uses every helper

/// Default absolute tolerance for oracle vs. kernel comparisons.
pub const DEFAULT_ABS_TOL: f32 = 1e-5;
/// Default relative tolerance for oracle vs. kernel comparisons.
pub const DEFAULT_REL_TOL: f32 = 1e-4;

/// Returns `true` iff `a` and `b` agree within the absolute-or-relative
/// tolerance contract used by this crate.
#[inline]
pub fn approx_eq(a: f32, b: f32) -> bool {
    approx_eq_tol(a, b, DEFAULT_ABS_TOL, DEFAULT_REL_TOL)
}

/// Same as [`approx_eq`] but with caller-supplied tolerances. Useful when
/// a kernel deliberately accepts more error than the contract default
/// (e.g. long RNNs).
#[inline]
pub fn approx_eq_tol(a: f32, b: f32, abs: f32, rel: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= abs {
        return true;
    }
    let scale = a.abs().max(b.abs());
    diff <= rel * scale
}

/// Tiny linear-congruential generator used everywhere a deterministic
/// stream of `f32` in `[0, 1)` is needed (router tie-break, diffusion
/// remask, mamba scan warmup, RWKV init, etc.).
///
/// The multiplier is the same one Knuth attributes to MMIX; the modulus
/// is `2^64`. The state never reaches zero for any nonzero seed and is
/// stable across platforms.
#[derive(Debug, Clone)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    /// Construct a generator from the given seed. `seed == 0` is mapped to
    /// `0xDEAD_BEEF_CAFE_BABE` so the generator is well-defined.
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0xDEAD_BEEF_CAFE_BABE
        } else {
            seed
        };
        Self { state }
    }

    /// Advance the state and return the next `u64`.
    pub fn next_u64(&mut self) -> u64 {
        // MMIX LCG: state = state * 6364136223846793005 + 1442695040888963407
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Return a uniformly distributed `f32` in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // Use top 24 bits so the value fits in an f32 mantissa.
        let v = (self.next_u64() >> 40) as u32;
        (v as f32) / (1u32 << 24) as f32
    }

    /// Return a uniformly distributed `f32` in `[-1, 1)`.
    pub fn next_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

/// In-place numerically-stable softmax over `xs`. After the call,
/// `xs[i]` is `exp(xs[i] - max) / sum(exp(xs[j] - max))`. Returns the
/// post-normalization sum (always `1.0` for finite inputs).
pub fn softmax_row(xs: &mut [f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // All -inf or NaN; leave the row as zeros.
        for x in xs.iter_mut() {
            *x = 0.0;
        }
        return 0.0;
    }
    let mut sum = 0.0f32;
    for x in xs.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in xs.iter_mut() {
            *x *= inv;
        }
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_eq_zero_matches_zero() {
        assert!(approx_eq(0.0, 0.0));
        assert!(approx_eq(1e-9, -1e-9));
    }

    #[test]
    fn approx_eq_relative_tolerance() {
        // 0.001 difference on values near 1000 -> within 1e-4 relative.
        assert!(approx_eq(1000.0, 1000.05));
        assert!(!approx_eq(1000.0, 1000.5));
    }

    #[test]
    fn lcg_is_deterministic_for_same_seed() {
        let mut a = Lcg::new(42);
        let mut b = Lcg::new(42);
        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn lcg_different_seeds_diverge() {
        let mut a = Lcg::new(1);
        let mut b = Lcg::new(2);
        let mut differed = false;
        for _ in 0..4 {
            if a.next_u64() != b.next_u64() {
                differed = true;
                break;
            }
        }
        assert!(differed);
    }

    #[test]
    fn lcg_zero_seed_is_well_defined() {
        let mut a = Lcg::new(0);
        // Should not panic and should produce finite f32s.
        for _ in 0..8 {
            let v = a.next_f32();
            assert!(v.is_finite());
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn lcg_next_signed_is_in_range() {
        let mut a = Lcg::new(7);
        for _ in 0..16 {
            let v = a.next_signed();
            assert!((-1.0..1.0).contains(&v));
        }
    }

    #[test]
    fn softmax_row_normalizes_to_unit_sum() {
        let mut xs = [1.0f32, 2.0, 3.0, 4.0];
        let sum = softmax_row(&mut xs);
        assert!((sum - 1.0).abs() < 1e-6);
        let s: f32 = xs.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_row_is_stable_for_large_inputs() {
        let mut xs = [1000.0f32, 1001.0, 1002.0];
        softmax_row(&mut xs);
        assert!(xs.iter().all(|x| x.is_finite()));
        let s: f32 = xs.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_row_empty_is_zero_sum() {
        let mut xs: [f32; 0] = [];
        assert_eq!(softmax_row(&mut xs), 0.0);
    }
}
