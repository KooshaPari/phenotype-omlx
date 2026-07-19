//! Discrete (masked) diffusion — *schedule* coverage (linear + cosine).
//!
//! This file is the **schedule** half of the discrete-diffusion test
//! family, split from the prior `discrete_diffusion.rs` (468L) in
//! turn-9's module-size sweep. The sampler half
//! (`discrete_diffusion_sampler.rs`) owns the shared types
//! (`SelectorMetadata`, `SelectorMode`, `StepKind`,
//! `DiscreteDiffusionOracle`, `lcg_next`, `ddm_key`, `ddm_registry`)
//! plus the three sampler-oracle tests. This file picks up only the
//! `Schedule` enum + `DiscreteDiffusionOracle` it needs (via
//! `pub(crate) use super::discrete_diffusion_sampler::{...}`) so
//! every step-test can call the oracle through the same code path.
//!
//! Two tests cover the schedule surface:
//!
//! 1. `ddm_step_respects_schedule` — schedule boundary invariant: at
//!    step `0` every position is masked (`alpha(0) = 1`, the noised
//!    prior); at step `num_steps` no position is masked
//!    (`alpha(N) = 0`, the clean data distribution). The oracle's
//!    re-mask probability derived from the schedule must respect
//!    these boundaries — the test pins both linear and cosine.
//! 2. `ddm_cosine_vs_linear_schedule_differs` — linear and cosine
//!    schedules must produce *different* mask counts for the same
//!    input, confirming the schedule parameter is wired through the
//!    oracle rather than ignored.

use super::discrete_diffusion_sampler::{DiscreteDiffusionOracle, Schedule};

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
