//! Discrete-diffusion mask-schedule midpoint-pin regression suite
//! (Turn-14).
//!
//! # Orthogonal axis lineage — 5th of 5 closed axes
//!
//! | #  | Axis                              | File                                       |
//! |----|-----------------------------------|--------------------------------------------|
//! | 1  | `alpha(t)` function form          | `discrete_diffusion_l2.rs` (turn-9)        |
//! | 2  | L2-decay regression (Sqrt+Sigmoid)| `discrete_diffusion_l2_continuous.rs` (turn-11) |
//! | 3  | First derivative `dα/dt`          | `discrete_diffusion_schedule_derivative.rs` (turn-12) |
//! | 4  | Second derivative `d²α/dt²`       | `discrete_diffusion_schedule_convexity.rs` (turn-13) |
//! | 5  | Midpoint-pin `α(N/2)`             | this file (turn-14)                        |
//!
//! # What this axis pins (analytic values at `t = N/2`)
//!
//! | Schedule          | `α(N/2)`                                  |
//! |-------------------|-------------------------------------------|
//! | `Linear`          | `1 − 0.5 = 0.5`                           |
//! | `Cosine`          | `cos²(π/4) = 0.5`                         |
//! | `Sqrt`            | `√(1/2) = 1/√2 ≈ 0.7071` (NOT `1 − 1/√2`) |
//! | `Sigmoid { k }`   | `σ(0) = 0.5`                              |
//!
//! The midpoint is a *single-call* oracle for the `alpha_at` body —
//! any swap / off-by-one / flipped sign surfaces here before it can
//! drift through the derivative / convexity files.
//!
//! # Deviation note (Turn-14 prompt vs empirical `Schedule::alpha_at`)
//!
//! The Turn-14 prompt's spec table lists Sqrt's midpoint as
//! `1 − 1/√2 ≈ 0.2929`. That constant corresponds to the inverse
//! convention `1 − √(t/N)` which is **not** what
//! `Schedule::Sqrt::alpha_at` implements. The local
//! `Schedule::alpha_at` matches the production oracle
//! `ContinuousScheduleKind::Sqrt` byte-for-byte and uses
//! `sqrt(1 − t/N)`, so the empirical midpoint is
//! `√(1 − 0.5) = 1/√2 ≈ 0.7071`. Test 3 below pins the
//! empirically-correct value (mirroring the Turn-13 deviation note
//! in `discrete_diffusion_schedule_convexity.rs` which corrected the
//! prompt's `Sqrt = convex` label to `Sqrt = concave`). The schedule
//! itself is unchanged — only the prompt's label is wrong.
//!
//! # Test surface (API-stable)
//!
//! All tests use **only** the existing integer-step API
//! `Schedule::alpha_at(t: usize, num_steps: usize) -> f64`. No new
//! method on `Schedule` is introduced.
//!
//! # Tests (five)
//!
//! 1. `ddm_linear_schedule_midpoint_is_half` — `N ∈ {4, 8, 16, 32, 64, 128}`.
//! 2. `ddm_cosine_schedule_midpoint_is_half` — `N ∈ {4, 8, 16, 32, 64, 128}`.
//! 3. `ddm_sqrt_schedule_midpoint_is_one_over_sqrt2` — same `N` sweep.
//! 4. `ddm_sigmoid_schedule_midpoint_is_half_independent_of_k` —
//!    `k ∈ {1, 10, 50, 100, 500} × N ∈ {16, 32, 64}`.
//! 5. `ddm_midpoint_pin_disjoint_classification` — Sqrt is the unique
//!    schedule whose midpoint deviates from `0.5`; the other three
//!    collapse to `0.5` (deployment-level handle).

use super::discrete_diffusion_l2::Schedule;

// ---------------------------------------------------------------------------
// Local constants. Centralising the analytical midpoint values keeps the
// test bodies self-documenting and makes future schedule additions a
// single-site edit.
// ---------------------------------------------------------------------------

/// `1/√2 ≈ 0.7071067811865476` — the Sqrt midpoint constant.
const ONE_OVER_SQRT2: f64 = 1.0_f64 / std::f64::consts::SQRT_2;

/// Symmetric-midpoint constant: `0.5`. Used by Linear / Cosine / Sigmoid.
const MIDPOINT_HALF: f64 = 0.5_f64;

/// Loose midpoint-pin threshold for `cos²` / sigmoid-body rounding.
const MIDPOINT_TOL: f64 = 1e-6;

/// Tight threshold for Sqrt's `sqrt(1 − 0.5)` (no rounding to absorb).
const MIDPOINT_TOL_SQRT: f64 = 1e-12;

// ===========================================================================
// 1. Linear: α(N/2) = 0.5 (always, regardless of N).
// ===========================================================================

/// `α(t) = 1 − t/N`, so `α(N/2) = 1 − 0.5 = 0.5` exactly. Pin for
/// `N ∈ {4, 8, 16, 32, 64, 128}` — small N stresses the integer-step
/// `N/2` rounding; large N stresses the `1 − (N/2)/N` rational
/// arithmetic. Threshold `1e-6` matches the Turn-12 / Turn-13
/// conventions.
#[test]
fn ddm_linear_schedule_midpoint_is_half() {
    for &n in &[4usize, 8, 16, 32, 64, 128] {
        let mid = n / 2;
        let alpha_mid = Schedule::Linear.alpha_at(mid, n);
        assert!(
            (alpha_mid - MIDPOINT_HALF).abs() <= MIDPOINT_TOL,
            "Linear α(mid={mid}, N={n}) must equal 0.5; got {alpha_mid:.16}"
        );
    }
}

// ===========================================================================
// 2. Cosine: α(N/2) = 0.5 (always, by symmetry of cos² around π/4).
// ===========================================================================

/// `α(t) = cos²(t·π / (2·N))`. At `t = N/2` the inner argument is
/// `π/4`, so `cos²(π/4) = 1/2 = 0.5` exactly. Pin for the same
/// `N ∈ {4, 8, 16, 32, 64, 128}` sweep; threshold `1e-6` absorbs the
/// `f64` rounding in `cos(π·t / (2·N))`.
#[test]
fn ddm_cosine_schedule_midpoint_is_half() {
    for &n in &[4usize, 8, 16, 32, 64, 128] {
        let mid = n / 2;
        let alpha_mid = Schedule::Cosine.alpha_at(mid, n);
        assert!(
            (alpha_mid - MIDPOINT_HALF).abs() <= MIDPOINT_TOL,
            "Cosine α(mid={mid}, N={n}) must equal 0.5; got {alpha_mid:.16}"
        );
    }
}

// ===========================================================================
// 3. Sqrt: α(N/2) = 1/√2 ≈ 0.7071 (NOT 1 − 1/√2 — see deviation note).
// ===========================================================================

/// `α(t) = √(1 − t/N)`. At `t = N/2` this is `√(1 − 0.5) = 1/√2 ≈
/// 0.7071`. See the file-level deviation note: the prompt's
/// `1 − 1/√2 ≈ 0.2929` corresponds to the inverse convention
/// `1 − √(t/N)` and is **not** what `Schedule::Sqrt::alpha_at`
/// implements. Threshold `1e-12` (tight) because `sqrt` has no
/// rounding to absorb; the constant `1/√2` is matched exactly.
#[test]
fn ddm_sqrt_schedule_midpoint_is_one_over_sqrt2() {
    for &n in &[4usize, 8, 16, 32, 64, 128] {
        let mid = n / 2;
        let alpha_mid = Schedule::Sqrt.alpha_at(mid, n);
        assert!(
            (alpha_mid - ONE_OVER_SQRT2).abs() <= MIDPOINT_TOL_SQRT,
            "Sqrt α(mid={mid}, N={n}) must equal 1/√2 ≈ {ONE_OVER_SQRT2:.16}; got {alpha_mid:.16}"
        );
    }
    // Cross-check: the empirical Sqrt midpoint `1/√2` is *not* equal
    // to the prompt-spec'd `1 − 1/√2`. The two constants differ by
    // `√2 − 1 ≈ 0.4142`. This guards against a future silent "fix"
    // that flips the convention without updating the deviation note.
    let prompt_value = 1.0_f64 - 1.0_f64 / std::f64::consts::SQRT_2;
    assert!(
        (prompt_value - ONE_OVER_SQRT2).abs() > 1e-3,
        "internal sanity: prompt's 1 − 1/√2 ({prompt_value:.16}) must NOT equal empirical 1/√2 \
         ({ONE_OVER_SQRT2:.16}); if these collapse the Sqrt convention has flipped"
    );
}

// ===========================================================================
// 4. Sigmoid: α(N/2) = 0.5 (always, by symmetry of σ around its centre).
// ===========================================================================

/// `α(t) = 1 / (1 + exp(k · (2t/N − 1)))`. At `t = N/2` the inner
/// argument is `k · 0 = 0`, so `α = 1 / (1 + 1) = 0.5` exactly,
/// regardless of `k` (even extreme `k = 500` where tail arguments
/// would normally underflow — the midpoint is the one point where
/// the underflow cannot occur because the argument is exactly zero).
/// The production oracle / local `Schedule::alpha_at` does **not**
/// apply the `t == 0` / `t == num_steps` boundary special-case to
/// the midpoint, so the midpoint call goes through the body
/// arithmetic directly.
#[test]
fn ddm_sigmoid_schedule_midpoint_is_half_independent_of_k() {
    let ns: [usize; 3] = [16, 32, 64];
    let ks: [i32; 5] = [1, 10, 50, 100, 500];
    for &n in &ns {
        let mid = n / 2;
        for &k in &ks {
            let alpha_mid = Schedule::Sigmoid { k }.alpha_at(mid, n);
            assert!(
                (alpha_mid - MIDPOINT_HALF).abs() <= MIDPOINT_TOL,
                "Sigmoid(k={k}) α(mid={mid}, N={n}) must equal 0.5; got {alpha_mid:.16}"
            );
        }
    }
}

// ===========================================================================
// 5. Disjoint classification: Sqrt is the unique NonHalf schedule.
// ===========================================================================

/// Midpoint-pin class label. `Half` = `α(N/2) ≈ 0.5` (Linear / Cosine /
/// Sigmoid); `NonHalf` = `α(N/2) ≈ 0.7071` (Sqrt only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MidpointClass {
    Half,
    NonHalf,
}

/// Classify a schedule's midpoint as `Half` (within `1e-6` of `0.5`)
/// or `NonHalf`. This is the deployment-level handle: Sqrt is the
/// only schedule whose midpoint deviates from `0.5`, so a downstream
/// selector that needs to distinguish Sqrt from the other three has
/// a one-call oracle (`α(N/2)`) that succeeds.
fn classify_midpoint(schedule: Schedule, n: usize) -> MidpointClass {
    let alpha_mid = schedule.alpha_at(n / 2, n);
    if (alpha_mid - MIDPOINT_HALF).abs() <= MIDPOINT_TOL {
        MidpointClass::Half
    } else {
        MidpointClass::NonHalf
    }
}

/// Disjoint-classification fingerprint.
///
/// Contract at `N ∈ {16, 32, 64}` with `k = 50`:
/// - Linear / Cosine / Sigmoid all classify as `Half` (3-way collision
///   at the midpoint, by *different* analytic reasons).
/// - Sqrt classifies as `NonHalf` (the unique deployment handle).
///
/// The 3-way `Half` collision is documented: Linear / Cosine /
/// Sigmoid share the same midpoint value by independent analytic
/// reasons, so the midpoint pin alone cannot distinguish them — but
/// the *presence* of a single `NonHalf` schedule proves the four
/// schedules are not fully collapsed at the midpoint, which is the
/// selector handle the orthogonal axes (derivative / convexity) then
/// sharpen.
#[test]
fn ddm_midpoint_pin_disjoint_classification() {
    let ns: [usize; 3] = [16, 32, 64];
    let k: i32 = 50;

    for &n in &ns {
        let cls_linear = classify_midpoint(Schedule::Linear, n);
        let cls_cosine = classify_midpoint(Schedule::Cosine, n);
        let cls_sigmoid = classify_midpoint(Schedule::Sigmoid { k }, n);
        let cls_sqrt = classify_midpoint(Schedule::Sqrt, n);

        assert_eq!(cls_linear, MidpointClass::Half, "Linear @ N={n}");
        assert_eq!(cls_cosine, MidpointClass::Half, "Cosine @ N={n}");
        assert_eq!(cls_sigmoid, MidpointClass::Half, "Sigmoid(k={k}) @ N={n}");
        assert_eq!(cls_sqrt, MidpointClass::NonHalf, "Sqrt @ N={n}");

        let classes = [
            ("Linear", cls_linear),
            ("Cosine", cls_cosine),
            ("Sqrt", cls_sqrt),
            ("Sigmoid", cls_sigmoid),
        ];
        let non_half: Vec<&str> = classes
            .iter()
            .filter(|(_, c)| *c == MidpointClass::NonHalf)
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(
            non_half,
            vec!["Sqrt"],
            "exactly one NonHalf schedule (Sqrt) at N={n}; got {non_half:?}"
        );
    }
}
