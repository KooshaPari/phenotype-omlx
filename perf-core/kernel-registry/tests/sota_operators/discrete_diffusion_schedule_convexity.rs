//! Discrete-diffusion mask-schedule convexity regression suite (Turn-13).
//!
//! # Orthogonal axis
//!
//! Locks the *second-derivative / convexity fingerprint* of every schedule
//! exposed by `Schedule::alpha_at`. Pairs with the Turn-12 first-derivative
//! file (schedule gradient) and the Turn-11 L2 file (schedule itself).
//! Integer-step central second-difference stencil is used throughout,
//! matching the Turn-12 finite-difference helpers:
//!
//! ```text
//!     d²α/dt² ≈ α(t+1) - 2·α(t) + α(t-1)         (unit step h = 1)
//! ```
//!
//! The interior range is `t ∈ [1, N-1]`. Boundary points (`t = 0`, `t = N`)
//! are not exercised (the central second-diff stencil is undefined there).
//!
//! # Schedule convexities (verified empirically against `Schedule::alpha_at`)
//!
//! | Schedule          | d²α/dt² signature                                              |
//! |-------------------|----------------------------------------------------------------|
//! | `Linear`          | exactly `0` everywhere                                         |
//! | `Sqrt`            | strictly negative on `(0, N)` (concave)                        |
//! | `Cosine`          | negative on `(0, N/2)`, `≈0` at `t = N/2`, positive on `(N/2, N)` |
//! | `Sigmoid { k }`   | negative on `(0, N/2)`, `≈0` at `t = N/2`, positive on `(N/2, N)` |
//!
//! Turn-13 deviation note: the Turn-13 prompt labels `Sqrt` as "convex",
//! `Cosine` as "globally concave", and `Sigmoid` as "convex-then-concave".
//! Those labels are inverted relative to the actual second derivatives
//! (and to the central second-diff stencil surface at every swept `N`).
//! The numerical contract pinned here matches the oracle and the analytic
//! calculus; the prompt's labels should be corrected in a follow-up.
//!
//! # Test plan (five orthogonal tests, mirroring Turn-12)
//!
//! 1. `Linear` second derivative is exactly zero (numerical floor 1e-6).
//! 2. `Sqrt` is strictly concave on the interior (sign-split across
//!    `(0, N)` must be all-negative).
//! 3. `Cosine` is concave on the first interior half and convex on the
//!    second interior half — verified by sign-split at `N/2`.
//! 4. `Sigmoid { k }` flips sign at the midpoint — strictly negative
//!    just before, ≈0 at, strictly positive just after.
//! 5. The four convexity fingerprints are mutually disjoint: each
//!    schedule's interior sign-sum pattern is unique.

#![allow(clippy::needless_range_loop)]

use super::discrete_diffusion_l2::Schedule;

// ---------------------------------------------------------------------------
// Finite-difference helpers (mirroring Turn-12's local-first-derivative
// helpers; integer-step central second-difference, unit h = 1).
// ---------------------------------------------------------------------------

/// Central second-difference at interior point `t ∈ [1, N-1]`.
/// Mirrors the Turn-12 `central_diff` helper style.
#[inline]
fn central_diff_2nd(schedule: Schedule, t: usize, num_steps: usize) -> f64 {
    debug_assert!(
        t > 0 && t < num_steps,
        "central_diff_2nd requires 0 < t < N"
    );
    let a_prev = schedule.alpha_at(t - 1, num_steps);
    let a_now = schedule.alpha_at(t, num_steps);
    let a_next = schedule.alpha_at(t + 1, num_steps);
    a_next - 2.0 * a_now + a_prev
}

/// Forward second-difference at `t = 0` (boundary-equivalent).
/// Provided for parity with Turn-12's helper triple; not exercised below.
#[allow(dead_code)]
#[inline]
fn forward_diff_2nd(schedule: Schedule, num_steps: usize) -> f64 {
    let a0 = schedule.alpha_at(0, num_steps);
    let a1 = schedule.alpha_at(1, num_steps);
    let a2 = schedule.alpha_at(2, num_steps);
    a2 - 2.0 * a1 + a0
}

/// Backward second-difference at `t = N` (boundary-equivalent).
#[allow(dead_code)]
#[inline]
fn backward_diff_2nd(schedule: Schedule, num_steps: usize) -> f64 {
    let a_nm2 = schedule.alpha_at(num_steps - 2, num_steps);
    let a_nm1 = schedule.alpha_at(num_steps - 1, num_steps);
    let a_n = schedule.alpha_at(num_steps, num_steps);
    a_n - 2.0 * a_nm1 + a_nm2
}

// ===========================================================================
// 1. Linear: second derivative vanishes.
// ===========================================================================

#[test]
fn ddm_linear_schedule_second_derivative_is_zero() {
    for &n in &[4usize, 8, 16, 32, 64, 128] {
        for t in 1..n {
            let d2 = central_diff_2nd(Schedule::Linear, t, n);
            assert!(
                d2.abs() <= 1e-6,
                "Linear d² at t={}, N={} should vanish, got {:.3e}",
                t,
                n,
                d2
            );
        }
    }
}

// ===========================================================================
// 2. Sqrt: strictly concave on (0, N).
// ===========================================================================

#[test]
fn ddm_sqrt_schedule_is_strictly_concave_on_interior() {
    for &n in &[4usize, 8, 32, 128] {
        for t in 1..n {
            let d2 = central_diff_2nd(Schedule::Sqrt, t, n);
            assert!(
                d2 < 0.0,
                "Sqrt d² at t={}, N={} must be strictly negative (concave), got {:.6e}",
                t,
                n,
                d2
            );
        }
    }
}

// ===========================================================================
// 3. Cosine: concave then convex (split at N/2).
// ===========================================================================

#[test]
fn ddm_cosine_schedule_is_concave_then_convex() {
    for &n in &[4usize, 8, 32, 128] {
        let mid = n / 2;
        // First interior half: t ∈ [1, mid-1] (skip when N=4 leaves mid=2 and
        // no first-half interior, since mid itself is the inflection).
        if mid >= 2 {
            for t in 1..mid {
                let d2 = central_diff_2nd(Schedule::Cosine, t, n);
                assert!(
                    d2 < 0.0,
                    "Cosine d² at t={}, N={} (first half) must be negative, got {:.6e}",
                    t,
                    n,
                    d2
                );
            }
        }
        // Second interior half: t ∈ [mid+1, N-1].
        for t in (mid + 1)..n {
            let d2 = central_diff_2nd(Schedule::Cosine, t, n);
            assert!(
                d2 > 0.0,
                "Cosine d² at t={}, N={} (second half) must be positive, got {:.6e}",
                t,
                n,
                d2
            );
        }
    }
}

// ===========================================================================
// 4. Sigmoid: concave then convex, zero at the midpoint.
// ===========================================================================

#[test]
fn ddm_sigmoid_schedule_changes_sign_at_midpoint() {
    for &n in &[16usize, 32, 64] {
        for &k in &[10i32, 50, 100] {
            let mid = n / 2;
            let sched = Schedule::Sigmoid { k };
            let d2_before = central_diff_2nd(sched, mid - 1, n);
            let d2_at = central_diff_2nd(sched, mid, n);
            let d2_after = central_diff_2nd(sched, mid + 1, n);

            assert!(
                d2_before < 0.0,
                "Sigmoid(k={}, N={}) d² at t=N/2-1 must be negative (concave), got {:.6e}",
                k,
                n,
                d2_before
            );
            // Inflection: at `t = N/2`, `α = 0.5`, analytic `d² = 0`.
            // The FD stencil at `t = N/2` reduces to `α(mid+1) - 1 + α(mid-1)`
            // which by the sigmoid's antisymmetry about the midpoint is
            // small but not identically zero for finite `k`. Floor is `1e-6`.
            assert!(
                d2_at.abs() < 1e-6,
                "Sigmoid(k={}, N={}) d² at t=N/2 must be ≈0, got {:.6e}",
                k,
                n,
                d2_at
            );
            assert!(
                d2_after > 0.0,
                "Sigmoid(k={}, N={}) d² at t=N/2+1 must be positive (convex), got {:.6e}",
                k,
                n,
                d2_after
            );
        }
    }
}

// ===========================================================================
// 5. Sign-classification fingerprints are mutually disjoint.
// ===========================================================================

/// Fingerprint label for the convexity class of a schedule (under the
/// integer-step central second-diff stencil).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fingerprint {
    /// All interior values ≈ 0 within a per-N floor (Linear).
    Zero,
    /// All interior values strictly negative (Sqrt: global concavity).
    StrictlyNegative,
    /// Sign changes: negative on first half, positive on second half
    /// (Cosine: peak magnitude at boundary, zero at midpoint).
    SignChangeBoundaryPeaked,
    /// Sign changes: negative on first half, positive on second half
    /// (Sigmoid: peak magnitude at midpoint, near-zero at boundaries).
    SignChangeMidpointPeaked,
}

fn classify_fingerprint(schedule: Schedule, n: usize) -> Fingerprint {
    assert!(n >= 4 && n % 2 == 0, "classify_fingerprint needs even N≥4");
    let mid = n / 2;
    // Maximum-magnitude interior point (for Zero-vs-nonzero disambiguation
    // and for the Cosine-vs-Sigmoid boundary/midpoint discriminator).
    let mut max_abs: f64 = 0.0;
    // Magnitude at the boundary (t = 1) and at the inflection-adjacent
    // (t = mid - 1): discriminates Cosine from Sigmoid for SignChange.
    let d2_boundary = central_diff_2nd(schedule, 1, n).abs();
    let d2_almost_mid = central_diff_2nd(schedule, mid - 1, n).abs();
    // Sign trackers over the interior.
    let mut all_negative = true;
    let mut has_sign_change = false;
    let mut last_sign: i8 = 0;
    for t in 1..n {
        let d2 = central_diff_2nd(schedule, t, n);
        let ad2 = d2.abs();
        if ad2 > max_abs {
            max_abs = ad2;
        }
        let s = if d2 > 0.0 {
            1
        } else if d2 < 0.0 {
            -1
        } else {
            0
        };
        if s > 0 {
            all_negative = false;
        }
        if last_sign != 0 && s != 0 && s != last_sign {
            has_sign_change = true;
        }
        if s != 0 {
            last_sign = s;
        }
    }
    // Linear's analytic d² is exactly 0; the FD surface returns a pure
    // rounding residual (~1e-16). The `1e-12` floor cleanly separates
    // the rounding residual from any schedule with non-trivial curvature
    // (Sqrt's smallest interior |d²| at N = 4 is ~2.5e-2).
    if max_abs <= 1e-12 {
        return Fingerprint::Zero;
    }
    if all_negative {
        return Fingerprint::StrictlyNegative;
    }
    if has_sign_change {
        // Distinguish Cosine (boundary magnitude dominates) from
        // Sigmoid (midpoint magnitude dominates).
        if d2_boundary > d2_almost_mid {
            Fingerprint::SignChangeBoundaryPeaked
        } else {
            Fingerprint::SignChangeMidpointPeaked
        }
    } else {
        // All positive — not used by any current schedule, but reserved
        // for symmetry. Kept for future expansion.
        Fingerprint::StrictlyNegative
    }
}

#[test]
fn ddm_convexity_sign_classification_is_disjoint() {
    let n = 32usize;
    let k = 50i32;
    let fp_linear = classify_fingerprint(Schedule::Linear, n);
    let fp_sqrt = classify_fingerprint(Schedule::Sqrt, n);
    let fp_cosine = classify_fingerprint(Schedule::Cosine, n);
    let fp_sigmoid = classify_fingerprint(Schedule::Sigmoid { k }, n);

    // Spot-check expected assignments.
    assert_eq!(fp_linear, Fingerprint::Zero, "Linear must classify as Zero");
    assert_eq!(
        fp_sqrt,
        Fingerprint::StrictlyNegative,
        "Sqrt must classify as StrictlyNegative"
    );
    assert_eq!(
        fp_cosine,
        Fingerprint::SignChangeBoundaryPeaked,
        "Cosine must classify as SignChangeBoundaryPeaked (peak |d²| at boundary)"
    );
    assert_eq!(
        fp_sigmoid,
        Fingerprint::SignChangeMidpointPeaked,
        "Sigmoid must classify as SignChangeMidpointPeaked (peak |d²| at midpoint)"
    );

    // The four fingerprints are mutually distinct.
    let fingerprints = [fp_linear, fp_sqrt, fp_cosine, fp_sigmoid];
    for i in 0..fingerprints.len() {
        for j in (i + 1)..fingerprints.len() {
            assert_ne!(
                fingerprints[i], fingerprints[j],
                "Schedules at indices ({}, {}) share fingerprint {:?}",
                i, j, fingerprints[i]
            );
        }
    }
}
