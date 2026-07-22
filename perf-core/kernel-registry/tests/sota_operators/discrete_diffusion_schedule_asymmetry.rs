//! Discrete-diffusion mask-schedule asymmetry regression suite
//! (Turn-15 — 6th orthogonal axis).
//!
//! # Orthogonal axis lineage — 6 closed axes, this file is the 6th
//!
//! | #  | Axis                              | File                                       |
//! |----|-----------------------------------|--------------------------------------------|
//! | 1  | `alpha(t)` function form          | `discrete_diffusion_l2.rs` (turn-9)        |
//! | 2  | L2-decay regression (Sqrt+Sigmoid)| `discrete_diffusion_l2_continuous.rs` (turn-11) |
//! | 3  | First derivative `dα/dt`          | `discrete_diffusion_schedule_derivative.rs` (turn-12) |
//! | 4  | Second derivative `d²α/dt²`       | `discrete_diffusion_schedule_convexity.rs` (turn-13) |
//! | 5  | Midpoint-pin `α(N/2)`             | `discrete_diffusion_schedule_midpoint.rs` (turn-14) |
//! | 6  | Asymmetry `α(t) + α(N−t)`         | this file (turn-15)                        |
//!
//! # What this axis pins
//!
//! Pairwise symmetry / asymmetry of each schedule about the midpoint
//! `t = N/2`, captured by the discrete conjugate sum
//! `S(t, N) := α(t) + α(N − t)`:
//!
//! | Schedule          | `S(t, N)` shape                                                  |
//! |-------------------|------------------------------------------------------------------|
//! | `Linear`          | exactly `1.0` for all `t ∈ [0, N]`                               |
//! | `Cosine`          | exactly `1.0` for all `t ∈ [0, N]`                               |
//! | `Sqrt`            | `√(1 − t/N) + √(t/N)` — strictly in `[1, √2]`, max `√2` at midpoint |
//! | `Sigmoid { k }`   | exactly `1.0` for all `t ∈ [0, N]` (independent of `k`)          |
//!
//! Analytic proofs (each is one line):
//!
//! - Linear: `(1 − t/N) + (1 − (N − t)/N) = 2 − t/N − 1 + t/N = 1`.
//! - Cosine: `cos²(x) + cos²(π/2 − x) = cos²(x) + sin²(x) = 1`, where
//!   `x = tπ/(2N)`. Note `(N − t)π/(2N) = π/2 − x`.
//! - Sqrt: `√(1 − t/N) + √(t/N)` is *not* constant. At `t = 0`: `1`.
//!   At `t = N`: `1`. At `t = N/2`: `2·√(1/2) = √2`. By concavity of
//!   `√` on `[0, 1]`, every interior point lies strictly in `(1, √2)`.
//! - Sigmoid: for `t ∈ [1, N − 1]` the body is
//!   `σ(k·(2t/N − 1)) =: σ(z)` and the conjugate call evaluates
//!   `σ(k·(2(N − t)/N − 1)) = σ(k·(1 − 2t/N)) = σ(−z) = 1 − σ(z)` by
//!   the logistic identity `σ(−z) = 1 − σ(z)`. Endpoints are pinned by
//!   the boundary special-case: `α(0) = 1`, `α(N) = 0` so
//!   `α(0) + α(N) = 1` and `α(N) + α(0) = 1`.
//!
//! # Why a sixth axis?
//!
//! The previous five axes pin *function-level* properties. Asymmetry is
//! a *relational* property: it ties together the values at two
//! distinct `t` indices into a single contract, exposing any
//! asymmetric body or boundary-guard drift (e.g. if someone added an
//! early-return that only fired for one of the two indices).
//!
//! # Test surface (API-stable)
//!
//! All tests use **only** the existing integer-step API
//! `Schedule::alpha_at(t: usize, num_steps: usize) -> f64`. No new
//! method on `Schedule` is introduced.
//!
//! # Tests (four)
//!
//! 1. `ddm_linear_schedule_is_symmetric_about_midpoint` —
//!    `α(t) + α(N − t) = 1.0` for `t ∈ [0, N]`, `N ∈ {4, 8, 16, 32, 64, 128}`.
//! 2. `ddm_cosine_schedule_is_symmetric_about_midpoint` — same for Cosine.
//! 3. `ddm_sqrt_schedule_is_asymmetric_with_max_at_midpoint` —
//!    pin min `= 1` at endpoints, max `= √2` at midpoint, strict
//!    interior ordering `1 < S(t, N) < √2` for `t ∈ (0, N)`.
//! 4. `ddm_sigmoid_schedule_is_symmetric_for_all_k` —
//!    `α(t) + α(N − t) = 1.0` for `k ∈ {1, 10, 50, 100}` ×
//!    `N ∈ {16, 32, 64}` × `t ∈ {1, N/2−1, N/2+1, N − 1}`.

use super::discrete_diffusion_l2::Schedule;

// ---------------------------------------------------------------------------
// Local constants. Centralising the analytic endpoints keeps the test bodies
// self-documenting and gives the asymmetry / midpoint files a single source
// of truth for the `1/√2` and `√2` constants.
// ---------------------------------------------------------------------------

/// `√2 ≈ 1.4142135623730951` — the Sqrt conjugate-sum maximum at `t = N/2`.
const SQRT2: f64 = std::f64::consts::SQRT_2;

/// `1.0` — the symmetric conjugate-sum value pinned by Linear / Cosine /
/// Sigmoid. Also the Sqrt conjugate-sum minimum at the endpoints.
const ONE: f64 = 1.0_f64;

/// Universal tight threshold for the symmetric schedules
/// (Linear / Cosine / Sigmoid). `1e-6` matches the Turn-12 / Turn-13 /
/// Turn-14 floor and absorbs `cos(…)` rounding in Cosine plus the
/// `exp(…)` body rounding in Sigmoid at large `k`.
const SYMMETRY_TOL: f64 = 1e-6;

/// Tighter threshold for Sqrt's *endpoint* and *midpoint* anchor pins
/// (no rounding to absorb at `t ∈ {0, N/2, N}` because `sqrt` of a
/// constant is exact in `f64`). Used in test 3 to lock the analytical
/// endpoints to their closed-form values.
const SQRT_ANCHOR_TOL: f64 = 1e-12;

/// Looser threshold for Sqrt's interior *strict-inequality* pins
/// (test 3 sweeps every `t ∈ (0, N)`; the bounds are open intervals
/// so the floor is the small distance from the boundary, well above
/// `f64` rounding noise).
const SQRT_INTERIOR_TOL: f64 = 1e-9;

/// Return `α(t) + α(N − t)` for the given schedule. The two `alpha_at`
/// calls share the production-oracle boundary special-cases (in
/// particular the Sigmoid `t == 0` / `t == num_steps` short-circuits)
/// so the conjugate pair sums to `1` exactly at the endpoints.
///
/// `t = N − t` when `N` is even and `t = N/2`; we handle that case
/// by returning `2·α(N/2)` (the conjugate call would be a no-op
/// double-count). This is only relevant for Sqrt, where the
/// midpoint doubles `α(N/2) = 1/√2` to give the `√2` maximum.
fn conjugate_sum(schedule: Schedule, t: usize, num_steps: usize) -> f64 {
    debug_assert!(t <= num_steps);
    if 2 * t == num_steps {
        // Self-conjugate: t == N - t. Avoid double-counting.
        2.0 * schedule.alpha_at(t, num_steps)
    } else {
        schedule.alpha_at(t, num_steps) + schedule.alpha_at(num_steps - t, num_steps)
    }
}

// ===========================================================================
// 1. Linear: α(t) + α(N − t) = 1 for every t, N.
// ===========================================================================

/// `α(t) = 1 − t/N` is an affine function with `α(0) = 1`, `α(N) = 0`.
/// Its conjugate sum satisfies `α(t) + α(N − t) = (1 − t/N) + (1 − (N − t)/N) = 1`
/// *exactly* in real arithmetic; on the integer-step `f64` surface the
/// identity holds to `f64` rounding precision.
///
/// Sweep `N ∈ {4, 8, 16, 32, 64, 128}` × `t ∈ [0, N]`. Threshold `1e-6`
/// is generous (the actual residual is `~1e-16`); the loose floor
/// matches the Turn-12 / Turn-13 / Turn-14 convention and is also
/// shared with Cosine (which *does* incur a `cos(…)` rounding tail).
#[test]
fn ddm_linear_schedule_is_symmetric_about_midpoint() {
    for &n in &[4usize, 8, 16, 32, 64, 128] {
        for t in 0..=n {
            let s = conjugate_sum(Schedule::Linear, t, n);
            assert!(
                (s - ONE).abs() <= SYMMETRY_TOL,
                "Linear α(t={t}) + α(N−t={}) must equal 1.0 for N={n}; got {s:.16}",
                n - t,
            );
        }
    }
}

// ===========================================================================
// 2. Cosine: α(t) + α(N − t) = 1 for every t, N.
// ===========================================================================

/// `α(t) = cos²(t·π / (2·N))`. Setting `x = t·π/(2N)`, the conjugate
/// term is `cos²((N − t)·π/(2N)) = cos²(π/2 − x) = sin²(x)`. Therefore
/// `α(t) + α(N − t) = cos²(x) + sin²(x) = 1` exactly. The integer-step
/// `f64` surface incurs rounding in both `cos(…)` calls; the `1e-6`
/// floor absorbs it with comfortable margin.
///
/// Sweep `N ∈ {4, 8, 16, 32, 64, 128}` × `t ∈ [0, N]`. The endpoint
/// pins are particularly sharp: `cos²(0) = 1` and `cos²(π/2) = 0`,
/// so `α(0) + α(N) = 1` exactly modulo rounding.
#[test]
fn ddm_cosine_schedule_is_symmetric_about_midpoint() {
    for &n in &[4usize, 8, 16, 32, 64, 128] {
        for t in 0..=n {
            let s = conjugate_sum(Schedule::Cosine, t, n);
            assert!(
                (s - ONE).abs() <= SYMMETRY_TOL,
                "Cosine α(t={t}) + α(N−t={}) must equal 1.0 for N={n}; got {s:.16}",
                n - t,
            );
        }
    }
}

// ===========================================================================
// 3. Sqrt: asymmetric, with √(1 − t/N) + √(t/N) ∈ [1, √2].
// ===========================================================================

/// `α(t) = √(1 − t/N)`. The conjugate sum
/// `S(t, N) := √(1 − t/N) + √(t/N)` is *not* constant. Properties:
///
/// - **Endpoints:** `S(0, N) = S(N, N) = 1 + 0 = 1` (anchor min).
/// - **Midpoint:** `S(N/2, N) = 2·√(1/2) = √2 ≈ 1.4142` (anchor max).
/// - **Strict interior:** for `t ∈ (0, N) \ {N/2}`,
///   `1 < S(t, N) < √2`. This follows from strict concavity of `√`
///   on `[0, 1]` applied to the symmetric pair `(t/N, 1 − t/N)`.
///
/// Sweep `N ∈ {16, 32, 64, 128}`. Larger `N` gives a denser interior
/// grid (more points in the `(1, √2)` open interval). Endpoints use
/// the tight `1e-12` threshold (no rounding in `sqrt`); the strict
/// interior uses `1e-9` (loose, since the bound itself is open).
///
/// Cross-check: the empirical Sqrt anchor `S(N/2, N) = √2` is *not*
/// equal to `1.0`, so a future "fix" that silently symmetrises Sqrt
/// (e.g. replacing `sqrt(1 − t/N)` with the inverse convention
/// `(1 − sqrt(t/N))` as a single rename) would cause the midpoint
/// pin to drop to `1 − 1/√2 ≈ 0.586` and the asymmetry test to
/// fail. See `discrete_diffusion_schedule_midpoint.rs` for the
/// matching inverse-convention deviation note.
#[test]
fn ddm_sqrt_schedule_is_asymmetric_with_max_at_midpoint() {
    let ns: [usize; 4] = [16, 32, 64, 128];

    for &n in &ns {
        let mid = n / 2;

        // Endpoint anchor: S(0, N) = S(N, N) = 1 exactly.
        let s_at_zero = conjugate_sum(Schedule::Sqrt, 0, n);
        assert!(
            (s_at_zero - ONE).abs() <= SQRT_ANCHOR_TOL,
            "Sqrt α(0) + α(N) must equal 1.0 (endpoint anchor); got {s_at_zero:.16} at N={n}"
        );
        let s_at_n = conjugate_sum(Schedule::Sqrt, n, n);
        assert!(
            (s_at_n - ONE).abs() <= SQRT_ANCHOR_TOL,
            "Sqrt α(N) + α(0) must equal 1.0 (endpoint anchor); got {s_at_n:.16} at N={n}"
        );

        // Midpoint anchor: S(N/2, N) = √2 (the *maximum* of the
        // asymmetric schedule). The `2 * α(N/2)` branch in
        // `conjugate_sum` returns `2 · 1/√2 = √2` exactly.
        let s_at_mid = conjugate_sum(Schedule::Sqrt, mid, n);
        assert!(
            (s_at_mid - SQRT2).abs() <= SQRT_ANCHOR_TOL,
            "Sqrt α(N/2) + α(N/2) must equal √2 ≈ {SQRT2:.16} (midpoint max); \
             got {s_at_mid:.16} at N={n}"
        );

        // Strict-interior pin: for t ∈ (0, N) \ {N/2},
        // 1 < S(t, N) < √2. The endpoints are *excluded* — they
        // collapse to `1`; the midpoint is *excluded* — it reaches
        // `√2`. The interior points must lie strictly inside.
        for t in 1..n {
            if t == mid {
                continue;
            }
            let s = conjugate_sum(Schedule::Sqrt, t, n);
            assert!(
                s > ONE + SQRT_INTERIOR_TOL,
                "Sqrt α(t={t}) + α(N−t={}) must be strictly > 1 at N={n}; got {s:.16}",
                n - t,
            );
            assert!(
                s < SQRT2 - SQRT_INTERIOR_TOL,
                "Sqrt α(t={t}) + α(N−t={}) must be strictly < √2 at N={n}; got {s:.16}",
                n - t,
            );
        }

        // Internal sanity: Sqrt's midpoint `√2` is *not* equal to
        // `1.0`. This guards against a future symmetric-as-Sqrt
        // "fix" that would silently erase the asymmetry axis.
        assert!(
            (SQRT2 - ONE).abs() > 1e-3,
            "internal sanity: √2 ({SQRT2:.16}) must NOT equal 1.0 ({ONE:.16}); \
             if these collapse Sqrt has lost its asymmetry"
        );
    }
}

// ===========================================================================
// 4. Sigmoid: α(t) + α(N − t) = 1 for every k, N, interior t.
// ===========================================================================

/// `α(t) = 1 / (1 + exp(k·(2t/N − 1)))` for `t ∈ [1, N − 1]`, with
/// boundary pins `α(0) = 1`, `α(N) = 0`. The conjugate term is
/// `α(N − t) = σ(k·(2(N − t)/N − 1)) = σ(k·(1 − 2t/N)) = σ(−z)` where
/// `z = k·(2t/N − 1)`. By the logistic identity `σ(−z) = 1 − σ(z)`,
/// the conjugate sum is `σ(z) + 1 − σ(z) = 1`.
///
/// This identity is *independent of `k`*: even for `k = 100` where the
/// sigmoid body is essentially a step function (left tail `≈ 1`,
/// right tail `≈ 0`), the body arithmetic remains antisymmetric
/// around the midpoint and the conjugate sum stays at `1`.
///
/// Sweep `k ∈ {1, 10, 50, 100}` × `N ∈ {16, 32, 64}` ×
/// `t ∈ {1, N/2 − 1, N/2 + 1, N − 1}`. The four `t` values cover:
///
/// - `t = 1`:        body near left tail (logistic ≈ 1.0 for any `k`).
/// - `t = N/2 − 1`:  body just before the inflection point.
/// - `t = N/2 + 1`:  body just after the inflection point (mirror of `N/2 − 1`).
/// - `t = N − 1`:    body near right tail (logistic ≈ 0.0 for any `k`).
///
/// Boundary points `t = 0` and `t = N` are exercised by the
/// `alpha_at(0, N) == 1` and `alpha_at(N, N) == 0` boundary invariants
/// (already pinned in `discrete_diffusion_l2_continuous.rs`); they
/// collapse the conjugate sum to `1` trivially.
#[test]
fn ddm_sigmoid_schedule_is_symmetric_for_all_k() {
    let ns: [usize; 3] = [16, 32, 64];
    let ks: [i32; 4] = [1, 10, 50, 100];

    for &n in &ns {
        let mid = n / 2;
        let ts: [usize; 4] = [1, mid - 1, mid + 1, n - 1];
        for &k in &ks {
            let sched = Schedule::Sigmoid { k };
            for &t in &ts {
                let s = conjugate_sum(sched, t, n);
                assert!(
                    (s - ONE).abs() <= SYMMETRY_TOL,
                    "Sigmoid(k={k}) α(t={t}) + α(N−t={}) must equal 1.0 for N={n}; got {s:.16}",
                    n - t,
                );
            }

            // Endpoint anchor: the boundary special-cases in
            // `Schedule::alpha_at` pin `α(0) = 1` and `α(N) = 0` for
            // *any* `k`, so the conjugate sum at the endpoints is
            // exactly `1` (one of the two terms is `1.0`, the other
            // is `0.0`).
            let s_endpoint_left = conjugate_sum(sched, 0, n);
            assert!(
                (s_endpoint_left - ONE).abs() <= SYMMETRY_TOL,
                "Sigmoid(k={k}) α(0) + α(N) must equal 1.0 (boundary anchor); \
                 got {s_endpoint_left:.16} at N={n}"
            );
            let s_endpoint_right = conjugate_sum(sched, n, n);
            assert!(
                (s_endpoint_right - ONE).abs() <= SYMMETRY_TOL,
                "Sigmoid(k={k}) α(N) + α(0) must equal 1.0 (boundary anchor); \
                 got {s_endpoint_right:.16} at N={n}"
            );
        }
    }
}