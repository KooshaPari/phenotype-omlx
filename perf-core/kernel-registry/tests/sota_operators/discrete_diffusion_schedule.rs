//! Discrete (masked) diffusion — *schedule* coverage (linear, cosine,
//! sqrt, sigmoid).
//!
//! This file is the **schedule** half of the discrete-diffusion test
//! family, split from the prior `discrete_diffusion.rs` (468L) in
//! turn-9's module-size sweep. The sampler half
//! (`discrete_diffusion_sampler.rs`) owns the shared types
//! (`SelectorMetadata`, `SelectorMode`, `StepKind`,
//! `DiscreteDiffusionOracle`, `lcg_next`, `ddm_key`, `ddm_registry`)
//! plus the three sampler-oracle tests. This file picks up the two
//! schedule types (`Schedule`, `ContinuousSchedule` +
//! `ContinuousScheduleKind`) and `DiscreteDiffusionOracle` it needs
//! (via `pub(crate) use super::discrete_diffusion_sampler::{...}`)
//! so every step-test can call the oracle through the same code path.
//!
//! `Schedule` is the original enum (`Linear` / `Cosine`); in turn-11
//! `ContinuousSchedule { kind, ... }` was added alongside it to
//! support arbitrary continuous noise schedules used by recent
//! SEDD / MDLM papers (`Sqrt`, `Sigmoid { k }`). The two share the
//! same `alpha_at(t, num_steps) -> f64` boundary invariant
//! (1.0 at t=0, 0.0 at t=num_steps) but live on separate types so
//! the byte-identical determinism of the original `Schedule` is
//! preserved.
//!
//! Five tests cover the schedule surface:
//!
//! 1. `ddm_step_respects_schedule` — schedule boundary invariant for
//!    `Linear`/`Cosine`: at step `0` every position is masked
//!    (`alpha(0) = 1`, the noised prior); at step `num_steps` no
//!    position is masked (`alpha(N) = 0`, the clean data
//!    distribution).
//! 2. `ddm_cosine_vs_linear_schedule_differs` — linear and cosine
//!    schedules must produce *different* mask counts for the same
//!    input.
//! 3. `ddm_alpha_at_boundaries_for_every_variant` — the same
//!    boundary invariant for all four `ContinuousScheduleKind`
//!    variants, including the new `Sqrt` and `Sigmoid { k: 10 }`.
//! 4. `ddm_sqrt_alpha_monotonic_decreasing` — `Sqrt` decays
//!    slower-than-linear at the start, so `alpha_at` must be
//!    monotone-nondecreasing-as-t-increases i.e. monotone-
//!    nonincreasing across the schedule domain.
//! 5. `ddm_sigmoid_alpha_decays_then_steepens` — `Sigmoid` with
//!    `k=10` has a sharper transition in the middle; at the
//!    quarter step the alpha must remain positive, and by the
//!    three-quarter step it must have dropped below 0.5.

use super::discrete_diffusion_sampler::{
    ContinuousSchedule, ContinuousScheduleKind, DiscreteDiffusionOracle, Schedule,
};

// ---------------------------------------------------------------------------
// Test 4 (post-split) — Schedule boundary invariant
// ---------------------------------------------------------------------------

/// Schedule boundary invariant: at step 0 every position is masked
/// (alpha(0) = 1, the noised prior); at step `num_steps` no position
/// is masked (alpha(N) = 0, the clean data distribution). The oracle's
/// re-mask probability derived from the schedule must respect these
/// boundaries — the test pins both linear and cosine.
#[test]
fn ddm_step_respects_schedule() {
    let num_steps: usize = 8;
    for schedule in [Schedule::Linear, Schedule::Cosine] {
        // Boundary check 1: alpha(0) == 1.
        let alpha_start = schedule.alpha_at(0, num_steps);
        assert!(
            (alpha_start - 1.0).abs() < 1e-9,
            "{schedule:?} alpha(0) must be 1.0 (fully-masked prior); got {alpha_start}"
        );

        // Boundary check 2: alpha(N) == 0.
        let alpha_end = schedule.alpha_at(num_steps, num_steps);
        assert!(
            alpha_end.abs() < 1e-9,
            "{schedule:?} alpha(num_steps) must be 0.0 (clean data); got {alpha_end}"
        );

        // Behavior check: feeding a fully-masked input at the last
        // step (`step = N - 1`) must yield a fully-masked output,
        // because the boundary re-mask probability is 1.
        let oracle = DiscreteDiffusionOracle::new(16, 4, num_steps, schedule);
        let n: usize = 32;
        let x_t: Vec<u32> = vec![oracle.mask_token_id; n];
        let mask: Vec<bool> = vec![true; n];
        let clean: Vec<u32> = (0..n).map(|i| ((i + 1) % (oracle.vocab_size as usize - 1)) as u32).collect();

        let out = oracle.step(&x_t, &mask, &clean, num_steps - 1, 1);
        assert_eq!(
            out,
            vec![oracle.mask_token_id; n],
            "{schedule:?} at the last step every position must be re-masked"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5 (post-split) — Schedule discriminator
// ---------------------------------------------------------------------------

/// Linear and cosine schedules must produce *different* mask counts
/// for the same input, confirming the schedule parameter is wired
/// through the oracle rather than ignored.
#[test]
fn ddm_cosine_vs_linear_schedule_differs() {
    let num_steps: usize = 16;
    let linear = DiscreteDiffusionOracle::new(32, 0, num_steps, Schedule::Linear);
    let cosine = DiscreteDiffusionOracle::new(32, 0, num_steps, Schedule::Cosine);
    let n: usize = 64;
    let x_t: Vec<u32> = vec![0; n]; // mask token id == 0
    let mask: Vec<bool> = vec![true; n];
    // Clean tokens live in 1..=31 to never collide with mask_token_id.
    let clean: Vec<u32> = (0..n).map(|i| ((i % 31) + 1) as u32).collect();

    // Pick a mid-range step where the two schedules diverge most.
    let step: usize = num_steps / 2;

    let linear_out = linear.step(&x_t, &mask, &clean, step, 99);
    let cosine_out = cosine.step(&x_t, &mask, &clean, step, 99);
    let linear_masked = linear.next_mask(&linear_out).iter().filter(|m| **m).count();
    let cosine_masked = cosine.next_mask(&cosine_out).iter().filter(|m| **m).count();

    assert_ne!(
        linear_masked, cosine_masked,
        "linear ({linear_masked}) and cosine ({cosine_masked}) schedules must produce different mask counts at step {step}/{num_steps}; \
         if they agree, the schedule parameter is being ignored"
    );
}

// ---------------------------------------------------------------------------
// Tests 6..10 (turn-11) — Continuous schedule surface (sqrt + sigmoid)
// ---------------------------------------------------------------------------

/// Helper: every `ContinuousScheduleKind` variant wrapped in a
/// `ContinuousSchedule`. Used by the boundary / monotonicity tests
/// below.
fn every_continuous_kind() -> [ContinuousSchedule; 4] {
    [
        ContinuousSchedule { kind: ContinuousScheduleKind::Linear },
        ContinuousSchedule { kind: ContinuousScheduleKind::Cosine },
        ContinuousSchedule { kind: ContinuousScheduleKind::Sqrt },
        ContinuousSchedule { kind: ContinuousScheduleKind::Sigmoid { k: 10 } },
    ]
}

/// The boundary invariant `alpha(0) == 1.0` and `alpha(num_steps) ==
/// 0.0` must hold for every `ContinuousScheduleKind` variant. This
/// pins the new `Sqrt` and `Sigmoid { k: 10 }` cases against the same
/// anchor that `Schedule::Linear` and `Schedule::Cosine` already obey.
#[test]
fn ddm_alpha_at_boundaries_for_every_variant() {
    let num_steps: usize = 32;
    for cs in every_continuous_kind() {
        let alpha_start = cs.alpha_at(0, num_steps);
        let alpha_end = cs.alpha_at(num_steps, num_steps);
        assert!(
            (alpha_start - 1.0).abs() < 1e-9,
            "{cs:?} alpha(0) must be 1.0; got {alpha_start}"
        );
        assert!(
            alpha_end.abs() < 1e-9,
            "{cs:?} alpha(num_steps) must be 0.0; got {alpha_end}"
        );
    }
}

/// `Sqrt` is the canonical slower-than-linear decay used by recent
/// SEDD / MDLM papers: at step = `num_steps / 2` the alpha must still
/// be strictly positive, and the sequence `alpha_at(t)` for
/// `t = 0..=num_steps` must be monotonically nonincreasing.
#[test]
fn ddm_sqrt_alpha_monotonic_decreasing() {
    let num_steps: usize = 16;
    let cs = ContinuousSchedule { kind: ContinuousScheduleKind::Sqrt };

    // Mid-step must remain positive (slower than linear).
    let mid = cs.alpha_at(num_steps / 2, num_steps);
    assert!(
        mid > 0.0,
        "Sqrt alpha({n}/2) must be > 0 (slower-than-linear decay); got {mid}",
        n = num_steps
    );

    // Sqrt(t/T) at t = T/2 is sqrt(1/2) ~= 0.707; pin a
    // tighter-than-trivial window so the test fails if the formula
    // accidentally becomes linear or cosine.
    assert!(
        (mid - 0.5_f64.sqrt()).abs() < 1e-9,
        "Sqrt alpha(N/2, N) must equal sqrt(1/2); got {mid}"
    );

    // Monotonicity: alpha_at is nonincreasing as t goes 0 -> N.
    let mut prev = cs.alpha_at(0, num_steps);
    for t in 1..=num_steps {
        let curr = cs.alpha_at(t, num_steps);
        assert!(
            curr <= prev + 1e-12,
            "Sqrt must be monotone nonincreasing; at t={t}/{num_steps} got {curr} > previous {prev}"
        );
        prev = curr;
    }
}

/// `Sigmoid { k: 10 }` is approximately 1.0 near t = 0, steepens
/// through the middle, and decays to ~0 at t = num_steps. At the
/// quarter step the alpha must still be positive (early-decay is
/// gentle), and by the three-quarter step it must already have
/// dropped below 0.5 (late-decay is steep).
#[test]
fn ddm_sigmoid_alpha_decays_then_steepens() {
    let num_steps: usize = 40;
    let cs = ContinuousSchedule { kind: ContinuousScheduleKind::Sigmoid { k: 10 } };

    let at_quarter = cs.alpha_at(num_steps / 4, num_steps);
    let at_three_quarters = cs.alpha_at(3 * num_steps / 4, num_steps);
    let at_mid = cs.alpha_at(num_steps / 2, num_steps);

    assert!(
        at_quarter > 0.0,
        "Sigmoid k=10 at t=N/4 must be > 0; got {at_quarter}"
    );
    assert!(
        at_three_quarters < 0.5,
        "Sigmoid k=10 at t=3N/4 must be < 0.5; got {at_three_quarters}"
    );
    // The sigmoid transition lives in the middle: between the
    // quarter and three-quarter steps the alpha must drop by at
    // least 0.25. This pins the "steepens through the middle"
    // shape independently of the specific k parameter.
    assert!(
        at_quarter - at_three_quarters > 0.25,
        "Sigmoid k=10 must drop by > 0.25 between t=N/4 and t=3N/4; \
         got quarter={at_quarter}, three-quarters={at_three_quarters}"
    );
    // Mid-step must land above the late-step alpha (the function is
    // monotone nonincreasing).
    assert!(
        at_mid + 1e-12 >= at_three_quarters,
        "Sigmoid k=10 must be monotone nonincreasing; got mid={at_mid} < three-quarters={at_three_quarters}"
    );
}

/// The discrete-diffusion oracle must remain byte-identical when run
/// twice with the same `(x_t, mask, clean, step, seed)` and a
/// `ContinuousSchedule { kind: Sqrt }`. This pins the determinism
/// contract for the new `Sqrt` variant.
#[test]
fn ddm_step_byte_identical_oracle_with_sqrt_schedule() {
    let cs = ContinuousSchedule { kind: ContinuousScheduleKind::Sqrt };
    let oracle = DiscreteDiffusionOracle::with_continuous(16, 4, 8, cs);
    let x_t: Vec<u32> = vec![4, 7, 4, 2, 4, 9, 4, 1]; // alternating mask + clean
    let mask: Vec<bool> = vec![true, false, true, false, true, false, true, false];
    let clean: Vec<u32> = vec![7, 7, 2, 2, 9, 9, 1, 1];

    let out_a = oracle.step(&x_t, &mask, &clean, 3, 0xC0FFEE);
    let out_b = oracle.step(&x_t, &mask, &clean, 3, 0xC0FFEE);
    let bytes_a: Vec<u8> = out_a.iter().flat_map(|t| t.to_le_bytes()).collect();
    let bytes_b: Vec<u8> = out_b.iter().flat_map(|t| t.to_le_bytes()).collect();
    assert_eq!(
        bytes_a, bytes_b,
        "oracle.step must be byte-identical under Sqrt schedule across calls with identical inputs"
    );
}

/// Same byte-identical determinism check for
/// `ContinuousSchedule { kind: Sigmoid { k: 10 } }`.
#[test]
fn ddm_step_byte_identical_oracle_with_sigmoid_schedule() {
    let cs = ContinuousSchedule { kind: ContinuousScheduleKind::Sigmoid { k: 10 } };
    let oracle = DiscreteDiffusionOracle::with_continuous(16, 4, 8, cs);
    let x_t: Vec<u32> = vec![4, 7, 4, 2, 4, 9, 4, 1];
    let mask: Vec<bool> = vec![true, false, true, false, true, false, true, false];
    let clean: Vec<u32> = vec![7, 7, 2, 2, 9, 9, 1, 1];

    let out_a = oracle.step(&x_t, &mask, &clean, 3, 0xC0FFEE);
    let out_b = oracle.step(&x_t, &mask, &clean, 3, 0xC0FFEE);
    let bytes_a: Vec<u8> = out_a.iter().flat_map(|t| t.to_le_bytes()).collect();
    let bytes_b: Vec<u8> = out_b.iter().flat_map(|t| t.to_le_bytes()).collect();
    assert_eq!(
        bytes_a, bytes_b,
        "oracle.step must be byte-identical under Sigmoid k=10 schedule across calls with identical inputs"
    );
}
