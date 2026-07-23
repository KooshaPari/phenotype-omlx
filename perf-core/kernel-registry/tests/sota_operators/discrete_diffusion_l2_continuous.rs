//! Discrete (masked) diffusion — *continuous schedule* L2-error
//! timestep-scaling tests (turn-11).
//!
//! This file is the **continuous-schedule** half of the L2-error
//! timestep-scaling test family, split out of
//! `discrete_diffusion_l2.rs` (696L after the turn-11 expansion)
//! to keep both files below the 500-line module cap. The legacy
//! half (`discrete_diffusion_l2.rs`) owns the [`Schedule`] enum
//! (Linear / Cosine / Sqrt / Sigmoid { k }) plus the
//! `reconstruction_l2_error` decoder, the `lcg_next` LCG helper,
//! the `DDM_T_SWEEP` constant, and the four linear/cosine L2-decay
//! tests. This file owns the eight turn-11 tests that exercise the
//! continuous `Sqrt` and `Sigmoid { k }` variants:
//!
//! 1. `discrete_diffusion_l2_reconstruction_error_monotonically_decays_with_T_sqrt` —
//!    L2-decay sweep under `Sqrt` (slower-than-linear).
//! 2. `discrete_diffusion_l2_reconstruction_error_monotonically_decays_with_T_sigmoid_k10` —
//!    L2-decay sweep under `Sigmoid { k: 10 }` (centred step).
//! 3. `discrete_diffusion_l2_error_below_clipping_floor_at_T_large_sqrt` —
//!    clipping-floor pin under `Sqrt` at the upper end of the sweep.
//! 4. `discrete_diffusion_l2_error_below_clipping_floor_at_T_large_sigmoid_k10` —
//!    clipping-floor pin under `Sigmoid { k: 10 }`.
//! 5. `discrete_diffusion_l2_error_at_T_fixed_under_seed_sqrt` —
//!    byte-identical determinism pin under `Sqrt`.
//! 6. `ddm_alpha_at_t_zero_for_every_continuous_schedule` —
//!    boundary invariant `alpha(0) == 1.0` across multiple
//!    `Sigmoid k` values plus a `Sqrt` spot-check.
//! 7. `ddm_alpha_at_t_num_steps_for_every_continuous_schedule` —
//!    companion test pinning `alpha(N) == 0.0`.
//! 8. `ddm_continuous_schedule_alpha_midpoint_relationships` —
//!    pins `Sqrt(N/2, N) == sqrt(1/2)` and
//!    `Sigmoid { k } (N/2, N) == 0.5` for the *body* of each
//!    formula (independent of the boundary guards).
//!
//! All decoding here is a pure scalar f32 oracle over `u32` token
//! ids: no MLX, no GPU, just arithmetic over small deterministic
//! tensors. The decoder is the same `reconstruction_l2_error` used
//! by the legacy half and lives there; this file imports it via
//! `super::discrete_diffusion_l2::{reconstruction_l2_error,
//! sweep_t_values, Schedule, ...}`.
//!
//! ## Why split
//!
//! Adding the eight turn-11 tests pushed
//! `discrete_diffusion_l2.rs` to 696L — well over the 500-line
//! module cap and almost 2× the 350-line target. Splitting the
//! turn-11 surface into its own file keeps each file focused on
//! one test family and brings both back under the 500-line cap
//! without changing the existing test surface at all.

use super::discrete_diffusion_l2::{reconstruction_l2_error, sweep_t_values, Schedule};

// ---------------------------------------------------------------------------
// Turn-11: continuous schedule L2-decay coverage (Sqrt + Sigmoid k=10)
// ---------------------------------------------------------------------------

/// `Sqrt` schedule sweep. Mirrors the linear/cosine sweeps in
/// `discrete_diffusion_l2.rs`: with more diffusion timesteps the
/// L2 reconstruction error must shrink by at least 2× between the
/// smallest `T` in `DDM_T_SWEEP` and the largest `T`. `Sqrt` decays
/// *slower* than linear at the start (alpha at mid-step is
/// `sqrt(1/2) ~= 0.707`, not `0.5`), so the per-step re-mask
/// probability near `t = num_steps / 2` is smaller under `Sqrt`
/// than under `Linear` — meaning a freshly-decoded position at
/// mid-step is less likely to be re-masked under `Sqrt` than under
/// `Linear`. This gives the decoder a *better* mid-schedule signal
/// and we still expect the same asymptotic collapse at `T > 200`.
/// Iterated programmatically from `DDM_T_SWEEP` so coverage can be
/// expanded without test-code edits.
#[test]
#[allow(non_snake_case)] // `T` is the diffusion timestep count, intentionally capital.
fn discrete_diffusion_l2_reconstruction_error_monotonically_decays_with_T_sqrt() {
    let tokens: [u32; 5] = [2, 11, 5, 7, 9];
    let vocab: u32 = 16;
    let mask_id: u32 = 4;
    let seed: u64 = 0xC0FFEE;

    let sweep = sweep_t_values();
    assert!(
        sweep.len() >= 2,
        "DDM_T_SWEEP must contain at least 2 entries to do an endpoint comparison; got {sweep:?}"
    );
    let t_min = *sweep.first().expect("non-empty sweep");
    let t_max = *sweep.last().expect("non-empty sweep");

    let l2_min: Vec<f64> = sweep
        .iter()
        .map(|&t| reconstruction_l2_error(t, Schedule::Sqrt, &tokens, vocab, mask_id, seed))
        .collect();
    let l2_t_min = l2_min[0];
    let l2_t_max = *l2_min.last().expect("non-empty L2 vector");

    assert!(
        l2_t_min > 2.0 * l2_t_max,
        "sqrt: L2 error at T={t_max} ({l2_t_max:.4}) must be < 0.5× L2 error at T={t_min} ({l2_t_min:.4}); \
         sweep T=[{t_min}..{t_max}] yielded L2={:?}",
        l2_min
    );
}

/// `Sigmoid { k: 10 }` schedule sweep. Mirrors the linear/cosine
/// sweeps in `discrete_diffusion_l2.rs`: with more diffusion
/// timesteps the L2 reconstruction error must shrink by at least
/// 2× between the smallest `T` in `DDM_T_SWEEP` and the largest
/// `T`. The sigmoid transition lives in the middle of the schedule:
/// the boundary re-mask probabilities at `t = 0` and
/// `t = num_steps - 1` are nearly identical to the linear schedule
/// (the sigmoid tails flatten out near the boundaries, so
/// `alpha(0) = 1.0` and `alpha(num_steps) = 0.0`), but the
/// mid-schedule re-mask probability is steeper than linear.
/// Iterated programmatically from `DDM_T_SWEEP` so coverage can be
/// expanded without test-code edits.
#[test]
#[allow(non_snake_case)] // `T` is the diffusion timestep count, intentionally capital.
fn discrete_diffusion_l2_reconstruction_error_monotonically_decays_with_T_sigmoid_k10() {
    let tokens: [u32; 5] = [2, 11, 5, 7, 9];
    let vocab: u32 = 16;
    let mask_id: u32 = 4;
    let seed: u64 = 0xC0FFEE;

    let sweep = sweep_t_values();
    assert!(
        sweep.len() >= 2,
        "DDM_T_SWEEP must contain at least 2 entries to do an endpoint comparison; got {sweep:?}"
    );
    let t_min = *sweep.first().expect("non-empty sweep");
    let t_max = *sweep.last().expect("non-empty sweep");

    let l2_min: Vec<f64> = sweep
        .iter()
        .map(|&t| {
            reconstruction_l2_error(
                t,
                Schedule::Sigmoid { k: 10 },
                &tokens,
                vocab,
                mask_id,
                seed,
            )
        })
        .collect();
    let l2_t_min = l2_min[0];
    let l2_t_max = *l2_min.last().expect("non-empty L2 vector");

    assert!(
        l2_t_min > 2.0 * l2_t_max,
        "sigmoid k=10: L2 error at T={t_max} ({l2_t_max:.4}) must be < 0.5× L2 error at T={t_min} ({l2_t_min:.4}); \
         sweep T=[{t_min}..{t_max}] yielded L2={:?}",
        l2_min
    );
}

/// Clipping-floor pair extension: at the largest `T` in
/// `DDM_T_SWEEP` the decoder's per-position bias
/// (`noise[i][t][clean[i]] = T`) strictly dominates the per-vocab
/// noise (`lcg_value % 200`), so the argmax is always the clean
/// token and the L2 reconstruction error collapses to the
/// floating-point floor. This is the turn-11 extension of
/// `discrete_diffusion_l2_error_below_clipping_floor_at_T_large`
/// to the continuous `Sqrt` schedule. The invariant is identical
/// — at `T > 200` the per-position bias dominates regardless of
/// which schedule is in use — but the test is split out so a
/// regression specifically in the `Sqrt` arm of the decoder can
/// be localised without wading through the linear/cosine arms.
#[test]
#[allow(non_snake_case)] // `T` is the diffusion timestep count, intentionally capital.
fn discrete_diffusion_l2_error_below_clipping_floor_at_T_large_sqrt() {
    let tokens: [u32; 5] = [2, 11, 5, 7, 9];
    let vocab: u32 = 16;
    let mask_id: u32 = 4;
    let seed: u64 = 0xC0FFEE;

    let sweep = sweep_t_values();
    let t_large = *sweep.last().expect("non-empty sweep");
    assert!(
        t_large > 200,
        "clipping-floor test requires t_large > 200 (the noise cap); got {t_large}"
    );

    let l2_sqrt = reconstruction_l2_error(t_large, Schedule::Sqrt, &tokens, vocab, mask_id, seed);

    assert!(
        l2_sqrt < 1e-9,
        "sqrt: L2 error at T={t_large} ({l2_sqrt}) must be < 1e-9 (clipping floor); \
         bias=T={t_large} > 200 noise cap, so every argmax should be the clean token"
    );
}

/// Clipping-floor pair extension: same invariant as
/// `discrete_diffusion_l2_error_below_clipping_floor_at_T_large_sqrt`
/// but for the `Sigmoid { k: 10 }` schedule. Split out so a
/// regression specifically in the `Sigmoid` arm of the decoder can
/// be localised without wading through the linear/cosine/sqrt arms.
#[test]
#[allow(non_snake_case)] // `T` is the diffusion timestep count, intentionally capital.
fn discrete_diffusion_l2_error_below_clipping_floor_at_T_large_sigmoid_k10() {
    let tokens: [u32; 5] = [2, 11, 5, 7, 9];
    let vocab: u32 = 16;
    let mask_id: u32 = 4;
    let seed: u64 = 0xC0FFEE;

    let sweep = sweep_t_values();
    let t_large = *sweep.last().expect("non-empty sweep");
    assert!(
        t_large > 200,
        "clipping-floor test requires t_large > 200 (the noise cap); got {t_large}"
    );

    let l2_sigmoid = reconstruction_l2_error(
        t_large,
        Schedule::Sigmoid { k: 10 },
        &tokens,
        vocab,
        mask_id,
        seed,
    );

    assert!(
        l2_sigmoid < 1e-9,
        "sigmoid k=10: L2 error at T={t_large} ({l2_sigmoid}) must be < 1e-9 (clipping floor); \
         bias=T={t_large} > 200 noise cap, so every argmax should be the clean token"
    );
}

/// Sqrt byte-identical determinism: mirrors
/// `discrete_diffusion_l2_error_at_T_fixed_under_seed` but under
/// the `Sqrt` schedule. The decoder is a pure deterministic
/// function of `(T, schedule, tokens, vocab, mask_id, seed)` —
/// running it twice at the same `T = 64` under the same seed must
/// yield byte-identical L2 errors. This is the universal "any
/// diffusion schedule must be deterministic under a fixed seed"
/// contract, now also pinned for the `Sqrt` arm.
#[test]
#[allow(non_snake_case)] // `T` is the diffusion timestep count, intentionally capital.
fn discrete_diffusion_l2_error_at_T_fixed_under_seed_sqrt() {
    let tokens: [u32; 5] = [2, 11, 5, 7, 9];
    let vocab: u32 = 16;
    let mask_id: u32 = 4;
    let seed: u64 = 0xC0FFEE;
    let num_steps: usize = 64;

    let l2_a = reconstruction_l2_error(num_steps, Schedule::Sqrt, &tokens, vocab, mask_id, seed);
    let l2_b = reconstruction_l2_error(num_steps, Schedule::Sqrt, &tokens, vocab, mask_id, seed);

    assert_eq!(
        l2_a.to_bits(),
        l2_b.to_bits(),
        "sqrt: L2 error must be bit-identical across two runs of the deterministic decode at \
         T={num_steps} under the same seed; got {l2_a} vs {l2_b}"
    );
}

// ---------------------------------------------------------------------------
// Turn-11: edge-case boundary tests for the continuous schedule surface
// ---------------------------------------------------------------------------

/// Boundary-invariant edge-case test (turn-11 forward-priority, see
/// `17_TURN_10_RESUME_NOTES.md` §9). Pins `alpha(0, N) == 1.0` for
/// every continuous schedule variant AND for several `Sigmoid k`
/// values — the boundary special-case branch in
/// `ContinuousSchedule::alpha_at` (the `if t == 0 { return 1.0; }`
/// guard in the oracle) is exactly what regresses if anyone
/// "simplifies" the sigmoid formula away. The companion test
/// `ddm_alpha_at_t_num_steps_for_every_continuous_schedule` pins
/// the `t == num_steps` end of the same branch.
///
/// The `Sigmoid k` sweep `[10, 50, 100]` covers (a) the production
/// default (`k = 10`), (b) a mid-range sharpness, and (c) a
/// near-step-function `k` that makes the floating-point rounding
/// errors in `exp(...)` near the boundary most pronounced. A
/// regression in the boundary guard would surface here as soon as
/// any one of these `k` values stops producing exactly `1.0`.
#[test]
fn ddm_alpha_at_t_zero_for_every_continuous_schedule() {
    let num_steps: usize = 32;
    let sigmoid_ks: [i32; 3] = [10, 50, 100];
    for &k in &sigmoid_ks {
        let sched = Schedule::Sigmoid { k };
        let alpha_start = sched.alpha_at(0, num_steps);
        assert!(
            (alpha_start - 1.0).abs() < 1e-12,
            "Sigmoid {{ k: {k} }} alpha(0, {num_steps}) must equal 1.0 (boundary guard); got {alpha_start}"
        );
    }
    // Spot-check the Sqrt arm too: Sqrt(0, N) == sqrt(1) == 1.0.
    let alpha_sqrt = Schedule::Sqrt.alpha_at(0, num_steps);
    assert!(
        (alpha_sqrt - 1.0).abs() < 1e-12,
        "Sqrt alpha(0, {num_steps}) must equal 1.0; got {alpha_sqrt}"
    );
}

/// Boundary-invariant edge-case test (turn-11 forward-priority,
/// companion to `ddm_alpha_at_t_zero_for_every_continuous_schedule`).
/// Pins `alpha(num_steps, N) == 0.0` for every continuous schedule
/// variant AND for several `Sigmoid k` values. The companion test
/// pins the `t == 0` end of the same
/// `if t == num_steps { return 0.0; }` guard in
/// `ContinuousSchedule::alpha_at`.
///
/// The `Sigmoid k` sweep `[10, 50, 100]` is identical to the
/// `t == 0` test so the two tests can be reasoned about together;
/// if either boundary slips, both tests will catch the regression
/// at the appropriate boundary.
#[test]
fn ddm_alpha_at_t_num_steps_for_every_continuous_schedule() {
    let num_steps: usize = 32;
    let sigmoid_ks: [i32; 3] = [10, 50, 100];
    for &k in &sigmoid_ks {
        let sched = Schedule::Sigmoid { k };
        let alpha_end = sched.alpha_at(num_steps, num_steps);
        assert!(
            alpha_end.abs() < 1e-12,
            "Sigmoid {{ k: {k} }} alpha({num_steps}, {num_steps}) must equal 0.0 (boundary guard); got {alpha_end}"
        );
    }
    // Spot-check the Sqrt arm too: Sqrt(N, N) == sqrt(0) == 0.0.
    let alpha_sqrt = Schedule::Sqrt.alpha_at(num_steps, num_steps);
    assert!(
        alpha_sqrt.abs() < 1e-12,
        "Sqrt alpha({num_steps}, {num_steps}) must equal 0.0; got {alpha_sqrt}"
    );
}

/// Midpoint relationship pin (turn-11 forward-priority). Pins the
/// midpoint values of the continuous schedules so a regression in
/// the *body* (not just the boundary guards) is caught:
///
/// - `Sqrt(N/2, N) == sqrt(1/2)` exactly. The `Sqrt` variant has
///   no boundary special-case at `t = N/2`, so the test pins the
///   body formula directly.
/// - `Sigmoid { k } (N/2, N) == 0.5` for any `k`. The sigmoid is
///   centred at `t = N/2` by construction: the `2*t/N - 1`
///   argument of `exp(...)` is exactly `0.0` at `t = N/2`, so the
///   sigmoid body evaluates to `1 / (1 + exp(0)) == 0.5`
///   regardless of `k`. We sweep `k ∈ {10, 50, 100}` to catch any
///   regression where someone introduces a `k`-dependent offset
///   to the centre.
///
/// If a regression shifts the sigmoid centre or the `Sqrt` body,
/// this test catches it independently of the boundary-guard tests
/// above.
#[test]
fn ddm_continuous_schedule_alpha_midpoint_relationships() {
    let num_steps: usize = 64;
    let mid = num_steps / 2;

    // Sqrt at mid-step must equal sqrt(1/2) within tight tolerance.
    let alpha_sqrt_mid = Schedule::Sqrt.alpha_at(mid, num_steps);
    assert!(
        (alpha_sqrt_mid - 0.5_f64.sqrt()).abs() < 1e-12,
        "Sqrt alpha({mid}, {num_steps}) must equal sqrt(1/2) (~0.7071); got {alpha_sqrt_mid}"
    );

    // Sigmoid at mid-step must equal 0.5 for any k — by construction
    // the sigmoid is centred at t = N/2.
    let sigmoid_ks: [i32; 3] = [10, 50, 100];
    for &k in &sigmoid_ks {
        let alpha_sig_mid = Schedule::Sigmoid { k }.alpha_at(mid, num_steps);
        assert!(
            (alpha_sig_mid - 0.5).abs() < 1e-12,
            "Sigmoid {{ k: {k} }} alpha({mid}, {num_steps}) must equal 0.5 (sigmoid midpoint); got {alpha_sig_mid}"
        );
    }
}
