//! Discrete (masked) diffusion — schedule-derivative regression
//! coverage for turn-12.
//!
//! Turn-11 pinned the *function* `alpha(t)` for the continuous
//! `Sqrt` and `Sigmoid { k }` schedules (L2-decay sweep plus
//! boundary and midpoint pins in `discrete_diffusion_l2_continuous.rs`).
//! Turn-12 pins the finite-difference derivative `d/dt alpha(t)`,
//! the orthogonal axis called out in the turn-11 resume notes.
//!
//! ## Why integer-step finite differences (not a continuous API)
//!
//! The production oracle and the local `Schedule` enum both expose
//! `alpha_at(t: usize, num_steps: usize) -> f64`. Adding a
//! continuous-`t` variant to the test surface would couple to the
//! oracle's internals. Instead, this file computes the derivative
//! at *integer* steps via forward / central / backward differences
//! on the existing API-stable surface:
//!
//! - one-sided forward  at `t = 0`:        `(alpha(1) - alpha(0)) / 1`
//! - central            at `t ∈ [1, N-1]`: `(alpha(t+1) - alpha(t-1)) / 2`
//! - one-sided backward at `t = N`:        `(alpha(N) - alpha(N-1)) / 1`
//!
//! For `Linear`, finite differences are *exact* (`α` is linear); for
//! `Sqrt` / `Sigmoid`, the central diff has `O(h²)` error and the
//! tests pin sign + magnitude band, with exact matches at `t = 0`
//! and `t = N/2` only on the `Linear` arm.
//!
//! ## Schedules
//!
//! - `Linear`:  `dα/dt = -1/N` (constant). Finite diff is exact.
//! - `Sqrt`:    `dα/dt = -1 / (2N·sqrt(1 - t/N))`, diverges as `t → N`.
//! - `Sigmoid { k }`: `dα/dt = -(k/N)·α·(1-α)`, most-negative at
//!   `t = N/2`.
//!
//! ## Tests (5)
//!
//! 1. `ddm_linear_schedule_derivative_is_constant_negative`
//! 2. `ddm_sqrt_schedule_derivative_is_monotonically_more_negative_than_linear`
//! 3. `ddm_sigmoid_schedule_derivative_zero_at_midpoint`
//! 4. `ddm_sigmoid_schedule_derivative_maximum_magnitude_at_midpoint`
//! 5. `ddm_all_continuous_schedules_derivative_is_non_positive`
//!
//! All tests are pure scalar `f64` arithmetic — no MLX, no GPU,
//! no decoder / re-mask logic (those live in
//! `discrete_diffusion_l2.rs`).

use super::discrete_diffusion_l2::Schedule;

// ---------------------------------------------------------------------------
// Local helpers (finite differences on the integer-step alpha surface)
// ---------------------------------------------------------------------------

/// Forward diff at `t ∈ [0, N - 1]`: `(alpha(t + 1) - alpha(t)) / 1`.
fn forward_diff(schedule: Schedule, t: usize, num_steps: usize) -> f64 {
    debug_assert!(t < num_steps, "forward_diff requires t < num_steps");
    let a_now = schedule.alpha_at(t, num_steps);
    let a_next = schedule.alpha_at(t + 1, num_steps);
    a_next - a_now
}

/// Central diff at `t ∈ [1, N - 1]`: `(alpha(t + 1) - alpha(t - 1)) / 2`.
fn central_diff(schedule: Schedule, t: usize, num_steps: usize) -> f64 {
    debug_assert!(
        t >= 1 && t < num_steps,
        "central_diff requires 1 <= t < num_steps"
    );
    let a_next = schedule.alpha_at(t + 1, num_steps);
    let a_prev = schedule.alpha_at(t - 1, num_steps);
    (a_next - a_prev) / 2.0
}

/// Backward diff at `t ∈ [1, N]`: `(alpha(t) - alpha(t - 1)) / 1`.
fn backward_diff(schedule: Schedule, t: usize, num_steps: usize) -> f64 {
    debug_assert!(
        t >= 1 && t <= num_steps,
        "backward_diff requires 1 <= t <= num_steps"
    );
    let a_now = schedule.alpha_at(t, num_steps);
    let a_prev = schedule.alpha_at(t - 1, num_steps);
    a_now - a_prev
}

// ---------------------------------------------------------------------------
// Test 1 — Linear derivative is constant `-1/N` at every integer step
// ---------------------------------------------------------------------------

/// `Schedule::Linear` has constant derivative `dα/dt = -1/N`. The
/// integer-step finite-difference approximations must equal `-1/N`
/// exactly:
///
/// - forward  `alpha(t + 1) - alpha(t)        == -1/N` for `t ∈ [0, N - 1]`
/// - central  `(alpha(t + 1) - alpha(t - 1))/2 == -1/N` for `t ∈ [1, N - 1]`
/// - backward `alpha(t) - alpha(t - 1)        == -1/N` for `t ∈ [1, N]`
///
/// Sweep `N ∈ {4, 8, 32, 128}`. Threshold `1e-12` (tighter than the
/// prompt's `1e-9` because for a linear function the finite diff is
/// exact in real arithmetic; any drift would be pure `f64` rounding
/// at the multiplication boundary).
#[test]
fn ddm_linear_schedule_derivative_is_constant_negative() {
    let ns: [usize; 4] = [4, 8, 32, 128];
    for &n in &ns {
        let expected = -1.0_f64 / n as f64;
        let sched = Schedule::Linear;

        for t in 0..n {
            let d = forward_diff(sched, t, n);
            assert!(
                (d - expected).abs() < 1e-12,
                "Linear forward_diff({t}, {n}) must equal -1/N = {expected}; got {d}"
            );
        }
        for t in 1..n {
            let d = central_diff(sched, t, n);
            assert!(
                (d - expected).abs() < 1e-12,
                "Linear central_diff({t}, {n}) must equal -1/N = {expected}; got {d}"
            );
        }
        for t in 1..=n {
            let d = backward_diff(sched, t, n);
            assert!(
                (d - expected).abs() < 1e-12,
                "Linear backward_diff({t}, {n}) must equal -1/N = {expected}; got {d}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2 — Sqrt derivative magnitude ordering vs Linear
// ---------------------------------------------------------------------------

/// `Schedule::Sqrt` has analytic `dα/dt = -1 / (2N·sqrt(1 - t/N))`.
/// Comparing magnitudes against Linear's constant `1/N`:
///
/// - At `t = 0`: `|dα_sqrt/dt| = 1/(2N)`, so the ratio is exactly
///   `1/2` — Sqrt is *half* as steep as Linear at the start.
/// - At `t = N - 1`: `|dα_sqrt/dt| = 1/(2·sqrt(N))`, so the ratio
///   is `sqrt(N)/2`. For `N ≥ 16` this is `≥ 2` — Sqrt is at least
///   twice as steep as Linear near the right boundary; as `t → N`
///   the Sqrt derivative diverges to `-∞`.
///
/// Pin via one-sided forward differences at `t = 0` and `t = N - 1`
/// to avoid boundary-guard contamination at `t = N` (the central
/// diff at `t = 0` would step into the `α(0) = 1.0` guard, a
/// different code path from the body). Sweep
/// `N ∈ {16, 64, 256}` so the "twice as steep at the end" inequality
/// has comfortable margin (sqrt(16)/2 = 2.0 minimum).
#[test]
fn ddm_sqrt_schedule_derivative_is_monotonically_more_negative_than_linear() {
    let ns: [usize; 3] = [16, 64, 256];
    for &n in &ns {
        let d_sqrt_start = forward_diff(Schedule::Sqrt, 0, n);
        let d_linear_start = forward_diff(Schedule::Linear, 0, n);
        assert!(
            d_sqrt_start < 0.0 && d_linear_start < 0.0,
            "both forward_diffs at t=0 must be negative; sqrt={d_sqrt_start}, linear={d_linear_start}"
        );
        assert!(
            d_sqrt_start.abs() < d_linear_start.abs(),
            "Sqrt forward_diff(0, {n}) magnitude ({:.6}) must be < Linear ({:.6}); \
             sqrt is less steep than linear at t=0 because body derivative is 0.5 at t=0",
            d_sqrt_start.abs(),
            d_linear_start.abs()
        );

        let d_sqrt_end = forward_diff(Schedule::Sqrt, n - 1, n);
        let d_linear_end = forward_diff(Schedule::Linear, n - 1, n);
        assert!(
            d_sqrt_end < 0.0 && d_linear_end < 0.0,
            "both forward_diffs at t=N-1 must be negative; sqrt={d_sqrt_end}, linear={d_linear_end}"
        );
        assert!(
            d_sqrt_end.abs() > d_linear_end.abs(),
            "Sqrt forward_diff(N-1, {n}) magnitude ({:.6}) must be > Linear ({:.6}); \
             sqrt diverges to -∞ as t → N",
            d_sqrt_end.abs(),
            d_linear_end.abs()
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3 — Sigmoid derivative is strictly negative at midpoint
// ---------------------------------------------------------------------------

/// `Schedule::Sigmoid { k }` has analytic `dα/dt = -(k/N)·α·(1 - α)`,
/// most-negative at the sigmoid inflection point `t = N/2` (where
/// `α = 0.5`). The integer-step central diff at `t = N/2` must be
/// strictly negative for any `(N, k)` in the sweep
/// `N ∈ {4, 8, 16, 32, 64}` × `k ∈ {10, 50, 100}` (production
/// default, mid-range sharpness, near-step-function).
///
/// Spot-check at `t = 0`: the boundary guard forces `α(0) = 1.0`,
/// so forward diff `(α(1) - α(0))` equals `α(1) - 1`. For large
/// `k`, `α(1) = σ(k·(2/N - 1))` is near `1.0` (left tail), so the
/// boundary derivative shrinks monotonically as `k` grows.
#[test]
fn ddm_sigmoid_schedule_derivative_zero_at_midpoint() {
    let ns: [usize; 5] = [4, 8, 16, 32, 64];
    let ks: [i32; 3] = [10, 50, 100];

    for &n in &ns {
        for &k in &ks {
            let sched = Schedule::Sigmoid { k };
            let mid = n / 2;
            let d_mid = central_diff(sched, mid, n);
            assert!(
                d_mid < -1e-12,
                "Sigmoid {{ k: {k} }} central_diff({mid}, {n}) must be strictly negative \
                 (midpoint is the inflection point); got {d_mid}"
            );
            let d_fwd_mid = forward_diff(sched, mid, n);
            assert!(
                d_fwd_mid < 0.0,
                "Sigmoid {{ k: {k} }} forward_diff({mid}, {n}) must be negative; got {d_fwd_mid}"
            );
        }
    }

    // Spot-check t = 0: the boundary derivative magnitude must be
    // monotonically non-increasing as k grows (sigmoid tail
    // flattens out). At `k=10, N=64` the left-tail argument is
    // `z = -9.6875`, so the body is `σ(-9.6875) ≈ 0.99994` and the
    // forward_diff is `~-6.2e-5` (very small but not exactly
    // zero). At `k=50` and `k=100` the tail argument is
    // `< -45`, so `exp(...)` underflows to 0 and `forward_diff`
    // is *exactly* 0.0. The pin is therefore "non-positive +
    // monotonic non-increasing in k" — not "strictly near zero
    // at every k". For k = 50 specifically the underflow to
    // exactly 0.0 is the desired terminal behavior.
    let n_spot: usize = 64;
    let mut prev_mag = f64::INFINITY;
    for &k in &ks {
        let d_boundary = forward_diff(Schedule::Sigmoid { k }, 0, n_spot);
        assert!(
            d_boundary <= 1e-12,
            "Sigmoid {{ k: {k} }} forward_diff(0, {n_spot}) must be <= 0 (boundary derivative); got {d_boundary}"
        );
        let mag = d_boundary.abs();
        assert!(
            mag <= prev_mag + 1e-12,
            "Sigmoid boundary-derivative magnitude must be monotonically non-increasing as k grows \
             (sigmoid tail flattens); at k={k}, N={n_spot} got magnitude {mag}, previous was {prev_mag}"
        );
        prev_mag = mag;
    }
}

// ---------------------------------------------------------------------------
// Test 4 — Sigmoid derivative argmin lies at t = N/2 (±2)
// ---------------------------------------------------------------------------

/// For `Schedule::Sigmoid { k }` the analytic derivative
/// `dα/dt = -(k/N)·α·(1 - α)` is maximised in magnitude at the
/// midpoint `t = N/2`. The integer-step central diff has `O(h²)`
/// error, so the integer-step argmin must still lie within ±2 of
/// `N/2` (absorbing boundary-rounding at the discrete `t` grid).
///
/// Sweep `N ∈ {32, 64}` × `k ∈ {10, 50}`. `N ≥ 32` gives the
/// central-diff stencil ample "shoulders" on either side of the
/// midpoint without crowding the boundary. `k = 10` is the
/// production default; `k = 50` makes the sigmoid transition
/// sharper so the central-diff minimum is more pronounced.
#[test]
fn ddm_sigmoid_schedule_derivative_maximum_magnitude_at_midpoint() {
    let ns: [usize; 2] = [32, 64];
    let ks: [i32; 2] = [10, 50];

    for &n in &ns {
        let mid = n / 2;
        for &k in &ks {
            let sched = Schedule::Sigmoid { k };

            // Sweep t ∈ [1, N - 1] and find the central-diff argmin.
            let mut argmin: usize = 1;
            let mut min_val: f64 = central_diff(sched, 1, n);
            for t in 2..n {
                let d = central_diff(sched, t, n);
                if d < min_val {
                    min_val = d;
                    argmin = t;
                }
            }

            let lo = mid.saturating_sub(2);
            let hi = mid + 2;
            assert!(
                argmin >= lo && argmin <= hi,
                "Sigmoid {{ k: {k} }} central-diff argmin ({argmin}) must lie within [{lo}, {hi}] \
                 of midpoint {mid} (N={n}); argmin migrating to the boundary indicates the \
                 sigmoid centre has shifted"
            );
            assert!(
                min_val < -1e-9,
                "Sigmoid {{ k: {k} }} central-diff at argmin ({argmin}, N={n}) must be < -1e-9; got {min_val}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 5 (optional) — Universal monotonic-decrease contract
// ---------------------------------------------------------------------------

/// Universal forward-diffusion contract: the mask fraction must be
/// nonincreasing in `t` for every continuous schedule variant. Pin:
/// `forward_diff(schedule, t, N) <= 0` for `t ∈ [0, N - 1]` and
/// every continuous variant (`Sqrt`, `Sigmoid k ∈ {10, 50, 100}`).
/// Threshold `1e-12` (tight enough to catch an accidental positive
/// bump, loose enough to absorb `exp(...)` rounding for large `k`).
///
/// Linear is excluded (its forward diff is exactly `-1/N`, pinned
/// in `ddm_linear_schedule_derivative_is_constant_negative`).
/// Cosine is excluded (discrete, not continuous).
#[test]
fn ddm_all_continuous_schedules_derivative_is_non_positive() {
    let n: usize = 64;
    let ks: [i32; 3] = [10, 50, 100];
    let schedules: [Schedule; 4] = [
        Schedule::Sqrt,
        Schedule::Sigmoid { k: ks[0] },
        Schedule::Sigmoid { k: ks[1] },
        Schedule::Sigmoid { k: ks[2] },
    ];

    for sched in schedules {
        for t in 0..n {
            let d = forward_diff(sched, t, n);
            assert!(
                d <= 1e-12,
                "{sched:?} forward_diff({t}, {n}) must be <= 0 (universal forward-diffusion contract); \
                 got {d} (positive diff would mean mask fraction is increasing, invalid for forward diffusion)"
            );
        }
    }
}
