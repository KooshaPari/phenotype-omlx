//! Discrete (masked) diffusion language model — *oracle* coverage.
//!
//! This file is the **oracle** half of the discrete-diffusion test
//! family, split out of `discrete_diffusion_sampler.rs` (407L, slightly
//! above the 350-line target) in turn-10. The split mirrors the natural
//! seam between *math/data* (the schedule, oracle, LCG helpers — kept
//! here) and *selector/registry* (the stub metadata, ddm_registry
//! wiring, selector-coverage test — moved to `discrete_diffusion_sampler.rs`).
//!
//! `discrete_diffusion_sampler.rs` re-exports every public item
//! (Schedule, DiscreteDiffusionOracle, ddm_key, ddm_registry, lcg_next)
//! and forwards test access via `pub(crate) use`s so the existing
//! callers (`discrete_diffusion_schedule.rs`, the per-tag coverage
//! matrix) remain byte-identical.
//!
//! Three tests own the oracle surface:
//!
//! 1. `ddm_step_byte_identical_to_oracle` — the reference oracle
//!    produces a byte-identical masked token tensor when called twice
//!    with the same `(x_t, mask, clean, step, seed)`. Pins the
//!    determinism contract for MDLM/D3PM-style decode.
//! 2. `ddm_masked_tokens_only_in_noised_positions` — only positions
//!    that were masked in the input may end up masked in the output.
//!    Pins the contract that the oracle never overwrites a clean
//!    position with `mask_token_id`.
//! 3. `ddm_step_uses_lcg_seed_mixing_for_seed_sweep` — sweeping the
//!    seed parameter across `(seed=1, 2, 3, ...)` produces distinct
//!    re-mask tensors, confirming the seed is wired through the LCG
//!    path rather than ignored.

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::{
    BackendKind, Capability, KernelKey, KernelRegistry,
};

use super::{
    build_record, make_candidate, samples_with_p95, shape, NOW_UNIX_MS,
    TEST_FINGERPRINT,
};

// ---------------------------------------------------------------------------
// Noise schedules (linear + cosine).
// ---------------------------------------------------------------------------

/// Noise schedule used by the discrete diffusion model. The schedule
/// controls how many positions stay masked at step `t` of `num_steps`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Schedule {
    /// Linear mask-fraction `alpha(t) = 1 - t / num_steps`.
    Linear,
    /// Cosine mask-fraction `alpha(t) = cos^2(t * pi / (2 * num_steps))`.
    /// Decays slower than linear at the start and faster at the end,
    /// so mid-step mask counts diverge from the linear schedule.
    Cosine,
}

impl Schedule {
    /// Mask fraction at step `t` in `0..=num_steps`. Returns a value
    /// in `[0.0, 1.0]`; the oracle uses it as a probability threshold
    /// for re-masking newly-unmasked tokens.
    pub(crate) fn alpha_at(self, t: usize, num_steps: usize) -> f64 {
        debug_assert!(t <= num_steps);
        let tn = t as f64 / num_steps as f64;
        match self {
            Schedule::Linear => (1.0 - tn).clamp(0.0, 1.0),
            Schedule::Cosine => {
                let c = (t as f64 * std::f64::consts::PI / (2.0 * num_steps as f64)).cos();
                (c * c).clamp(0.0, 1.0)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Continuous schedules (turn-11).
//
// Generalizes the legacy `Schedule` enum to support arbitrary continuous
// noise schedules used by recent SEDD / MDLM papers (`Sqrt`,
// `Sigmoid { k }`). The boundary invariant is identical:
// `alpha(0, N) == 1.0` and `alpha(N, N) == 0.0`. The legacy
// `Schedule` enum is kept untouched so any byte-identical determinism
// pinned against it remains stable.
// ---------------------------------------------------------------------------

/// Family discriminator for [`ContinuousSchedule`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContinuousScheduleKind {
    /// Linear mask-fraction `alpha(t) = 1 - t / num_steps`. Matches
    /// the legacy [`Schedule::Linear`] byte-for-byte.
    Linear,
    /// Cosine mask-fraction `alpha(t) = cos^2(t * pi / (2 * num_steps))`.
    /// Matches the legacy [`Schedule::Cosine`] byte-for-byte.
    Cosine,
    /// Sqrt mask-fraction `alpha(t) = sqrt(1 - t / num_steps)`.
    /// Slower-than-linear decay: at the mid-step alpha is
    /// `sqrt(1/2) ~= 0.707`, not `0.5`. Used by SEDD-style absorption
    /// schedules.
    Sqrt,
    /// Sigmoid mask-fraction `alpha(t) = 1 / (1 + exp(k * (2*t/N - 1)))`,
    /// centred at `t = N/2` with steepness `k`. Larger `k` yields a
    /// sharper transition through the middle and gentler tails
    /// near the boundaries. Used by MDLM-style reparameterised
    /// schedules.
    Sigmoid { k: i32 },
}

/// Wrapper struct so callers can pass a `ContinuousSchedule { kind, ... }`
/// to the oracle without reaching into the kind enum directly. Keeping
/// the wrapper distinct from the legacy [`Schedule`] enum preserves
/// the byte-identical determinism of the original test surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContinuousSchedule {
    pub(crate) kind: ContinuousScheduleKind,
}

impl ContinuousSchedule {
    /// Mask fraction at step `t` in `0..=num_steps`. Returns a value
    /// in `[0.0, 1.0]`; the oracle uses it as a probability threshold
    /// for re-masking newly-unmasked tokens. The boundary invariant
    /// is identical to [`Schedule::alpha_at`]: `alpha(0) = 1.0` and
    /// `alpha(num_steps) = 0.0` for every variant.
    pub(crate) fn alpha_at(self, t: usize, num_steps: usize) -> f64 {
        debug_assert!(t <= num_steps);
        let tn = t as f64 / num_steps as f64;
        match self.kind {
            ContinuousScheduleKind::Linear => (1.0 - tn).clamp(0.0, 1.0),
            ContinuousScheduleKind::Cosine => {
                let c = (t as f64 * std::f64::consts::PI / (2.0 * num_steps as f64)).cos();
                (c * c).clamp(0.0, 1.0)
            }
            ContinuousScheduleKind::Sqrt => (1.0 - tn).max(0.0).sqrt(),
            ContinuousScheduleKind::Sigmoid { k } => {
                // The boundary values must be exactly 1.0 / 0.0 to
                // match the shared boundary invariant; clamp the
                // endpoints before evaluating the sigmoid.
                if t == 0 {
                    return 1.0;
                }
                if t == num_steps {
                    return 0.0;
                }
                // 1 / (1 + exp(k * (2*t/N - 1))). At t = 0 the
                // exponent is -k → alpha ~= 1.0 for large k; at t = N
                // the exponent is +k → alpha ~= 0.0 for large k.
                let kf = k as f64;
                let z = kf * (2.0 * tn - 1.0);
                1.0 / (1.0 + z.exp())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reference oracle.
//
// The oracle is the test's source of truth: given a *noised* token
// sequence and the *clean* sequence, the model output is the one-hot
// of the clean token at every position. This is the standard MDLM /
// D3PM simplification that lets us pin byte-identical behavior
// without training a real masked LM.
// ---------------------------------------------------------------------------

/// Discrete-diffusion reference oracle. Mirrors the parameters the
/// runtime will eventually hand to a real MDLM-style model:
/// `vocab_size`, `mask_token_id`, `num_steps`, and `schedule`.
///
/// Internally the schedule is stored as a [`ContinuousSchedule`] so a
/// single `step` path covers both the legacy [`Schedule`] enum and
/// the newer `Sqrt`/`Sigmoid` variants. The legacy `new(...)`
/// constructor bridges the legacy enum into the continuous form,
/// preserving the byte-identical determinism of every pre-turn-11
/// test.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DiscreteDiffusionOracle {
    pub(crate) vocab_size: u32,
    pub(crate) mask_token_id: u32,
    pub(crate) num_steps: usize,
    pub(crate) schedule: ContinuousSchedule,
}

impl DiscreteDiffusionOracle {
    pub(crate) fn new(vocab_size: u32, mask_token_id: u32, num_steps: usize, schedule: Schedule) -> Self {
        // The mask token must be a valid vocab id so it round-trips
        // through the buffer byte representation.
        assert!(
            mask_token_id < vocab_size,
            "mask_token_id ({mask_token_id}) must be < vocab_size ({vocab_size})"
        );
        let continuous_kind = match schedule {
            Schedule::Linear => ContinuousScheduleKind::Linear,
            Schedule::Cosine => ContinuousScheduleKind::Cosine,
        };
        Self {
            vocab_size,
            mask_token_id,
            num_steps,
            schedule: ContinuousSchedule { kind: continuous_kind },
        }
    }

    /// Construct an oracle around an arbitrary
    /// [`ContinuousSchedule`]. Used by the newer `Sqrt`/`Sigmoid`
    /// schedule tests; the legacy `Linear`/`Cosine` schedules can also
    /// be passed via [`ContinuousScheduleKind::Linear`] / [`Cosine`]
    /// and produce the same step outputs as `new(...)`.
    pub(crate) fn with_continuous(
        vocab_size: u32,
        mask_token_id: u32,
        num_steps: usize,
        schedule: ContinuousSchedule,
    ) -> Self {
        assert!(
            mask_token_id < vocab_size,
            "mask_token_id ({mask_token_id}) must be < vocab_size ({vocab_size})"
        );
        Self {
            vocab_size,
            mask_token_id,
            num_steps,
            schedule,
        }
    }

    /// Re-mask policy: a freshly-decoded position is re-masked with
    /// probability `1 - alpha(t) / alpha(t-1)`, i.e. the schedule
    /// delta. This matches the MDLM formulation in Austin et al.
    /// (2023) §3.1 and keeps the test oracle deterministic.
    ///
    /// `step` is the current step in `0..num_steps`. `seed` is
    /// mixed into the LCG so two distinct invocations of the oracle
    /// at the same step still produce the same output given the
    /// same `(x_t, clean)` inputs. At `step == num_steps - 1` the
    /// schedule reaches `alpha(num_steps) = 0`, which means
    /// every newly-decoded position is re-masked — the fully-noised
    /// prior is restored at the boundary.
    pub(crate) fn step(&self, x_t: &[u32], mask: &[bool], clean: &[u32], step: usize, seed: u64) -> Vec<u32> {
        debug_assert_eq!(x_t.len(), clean.len());
        debug_assert_eq!(mask.len(), clean.len());
        debug_assert!(step < self.num_steps);

        // 1. Decode every currently-masked position by taking the
        //    one-hot of the clean token at that position (oracle).
        //    Positions that are not masked are left alone.
        let mut next_x: Vec<u32> = x_t.to_vec();
        for (i, &m) in mask.iter().enumerate() {
            if m {
                next_x[i] = clean[i];
            }
        }

        // 2. Re-mask a fraction of the *newly-decoded* positions
        //    according to the schedule delta. Re-masking only
        //    applies to positions that were masked in the input —
        //    positions that were already clean in the input must
        //    never be re-masked (they were not produced by the
        //    denoiser).
        let alpha_prev = self.schedule.alpha_at(step, self.num_steps);
        let alpha_now = self.schedule.alpha_at(step + 1, self.num_steps);
        // At the last step the schedule reaches zero: the boundary
        // re-masks every position. Avoid 0/0 by special-casing.
        let re_mask_prob = if alpha_prev <= f64::EPSILON {
            1.0
        } else {
            ((alpha_prev - alpha_now) / alpha_prev).clamp(0.0, 1.0)
        };
        // Threshold against the LCG's low 24 bits (range 0..2^24) so
        // the threshold is comparable to a 0..1 probability. Using
        // the raw 64-bit output would make `scaled < 1e6` match
        // almost never — this is the most common LCG bug.
        let scaled = (re_mask_prob * 0xFF_FFFF as f64).round() as u64;

        for (i, &m_in) in mask.iter().enumerate() {
            if !m_in {
                continue;
            }
            let lcg = lcg_next(seed.wrapping_add(step as u64).wrapping_add(i as u64));
            if (lcg & 0xFF_FFFF) < scaled {
                next_x[i] = self.mask_token_id;
            }
        }

        next_x
    }

    /// Build the next-step mask tensor: true where the position is
    /// masked (either newly re-masked or still masked from a prior
    /// step), false where decoded.
    pub(crate) fn next_mask(&self, x_next: &[u32]) -> Vec<bool> {
        x_next.iter().map(|&t| t == self.mask_token_id).collect()
    }
}

/// Tiny linear-congruential generator (MMIX constants). Deterministic,
/// seed-stable, and avoids pulling in `rand`.
pub(crate) fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ---------------------------------------------------------------------------
// Registry wiring (used by sampler tests + the schedule file).
// ---------------------------------------------------------------------------

pub(crate) fn ddm_key(vocab_size: usize, num_steps: usize) -> KernelKey {
    // vocab = m=vocab_size, total_steps = n=num_steps.
    KernelKey {
        operator_kind: OperatorKind::DiscreteDiffusion,
        attention_kind: None,
        shape_signature: shape(vocab_size, num_steps, vocab_size, 1, 1, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

pub(crate) fn ddm_registry() -> (KernelRegistry, kernel_registry::CandidateId, kernel_registry::CandidateId) {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(128, 64, 128, 4, 1, 1);
    let scalar = make_candidate(
        "DdmScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "DdmMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_scalar = scalar.id;
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = ddm_key(16, 8);
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_scalar, key.clone(), &samples_with_p95(9500), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(2100), Some(NOW_UNIX_MS + 86_400_000)),
    );
    (reg, id_scalar, id_metal)
}

// ---------------------------------------------------------------------------
// Tests (oracle surface).
// ---------------------------------------------------------------------------

/// The reference oracle produces a byte-identical masked token tensor
/// when called twice with the same `(x_t, mask, clean, step, seed)`.
/// This pins the determinism contract for MDLM/D3PM-style decode.
#[test]
fn ddm_step_byte_identical_to_oracle() {
    let oracle = DiscreteDiffusionOracle::new(16, 4, 8, Schedule::Linear);
    let x_t: Vec<u32> = vec![4, 7, 4, 2, 4, 9, 4, 1]; // alternating mask + clean
    let mask: Vec<bool> = vec![true, false, true, false, true, false, true, false];
    let clean: Vec<u32> = vec![7, 7, 2, 2, 9, 9, 1, 1];

    let out_a = oracle.step(&x_t, &mask, &clean, 3, 0xC0FFEE);
    let out_b = oracle.step(&x_t, &mask, &clean, 3, 0xC0FFEE);
    let bytes_a: Vec<u8> = out_a.iter().flat_map(|t| t.to_le_bytes()).collect();
    let bytes_b: Vec<u8> = out_b.iter().flat_map(|t| t.to_le_bytes()).collect();
    assert_eq!(
        bytes_a, bytes_b,
        "oracle.step must be byte-identical across calls with identical inputs"
    );

    // And the next-mask tensor derived from the output must agree
    // exactly with the one derived from the second invocation.
    let mask_a = oracle.next_mask(&out_a);
    let mask_b = oracle.next_mask(&out_b);
    assert_eq!(mask_a, mask_b);
}

/// Only positions that were masked in the input may end up masked in
/// the output. A non-masked position that contains a real token id
/// must never be overwritten with `mask_token_id` by the oracle.
#[test]
fn ddm_masked_tokens_only_in_noised_positions() {
    let oracle = DiscreteDiffusionOracle::new(8, 5, 4, Schedule::Linear);
    // x_t carries known tokens at every position; positions [0, 2, 5]
    // are flagged as masked (these are the *noised* positions).
    let x_t: Vec<u32> = vec![2, 3, 6, 1, 0, 4, 3, 7];
    let mask: Vec<bool> = vec![true, false, true, false, false, true, false, false];
    let clean: Vec<u32> = vec![3, 3, 1, 1, 0, 7, 3, 7];

    let out = oracle.step(&x_t, &mask, &clean, 1, 42);
    let next_mask = oracle.next_mask(&out);

    // Every position that was masked in the input must have a
    // well-defined output state: either still masked (re-masked) or
    // decoded to the clean token. No new masks should appear at
    // positions that were clean in the input.
    for (i, (&m_in, &m_out)) in mask.iter().zip(next_mask.iter()).enumerate() {
        if !m_in {
            // Clean input positions: the oracle may decode them to
            // their own clean token (== x_t[i] here) but must never
            // turn them into the mask token.
            assert_eq!(
                out[i], x_t[i],
                "non-masked input position {i} must not be overwritten with the mask token"
            );
            assert!(
                !m_out,
                "non-masked input position {i} must not appear masked in the output"
            );
        } else {
            // Noised position: output is either mask_token_id or clean[i].
            assert!(
                out[i] == oracle.mask_token_id || out[i] == clean[i],
                "noised position {i} must decode to clean[i] or stay masked, got {}",
                out[i]
            );
        }
    }
}

/// Sweeping the seed parameter across multiple values produces
/// distinct re-mask tensors at the boundary step. Confirms the seed
/// is wired through the LCG path rather than ignored. At step 0 with
/// a fully-masked input and `schedule = Cosine`, alpha_prev = 1.0
/// and alpha_now = cos^2(pi / (2 * num_steps)) < 1.0, so the
/// re-mask probability is `(1.0 - alpha_now) / 1.0 = 1.0 - alpha_now`,
/// strictly between 0 and 1 — the LCG seed actually matters.
#[test]
fn ddm_step_uses_lcg_seed_mixing_for_seed_sweep() {
    let oracle = DiscreteDiffusionOracle::new(64, 0, 4, Schedule::Cosine);
    let n: usize = 32;
    let x_t: Vec<u32> = vec![0; n]; // mask token id == 0
    let mask: Vec<bool> = vec![true; n];
    // Clean tokens live in 1..=63 to never collide with mask_token_id.
    let clean: Vec<u32> = (0..n).map(|i| ((i % 63) + 1) as u32).collect();

    // step = 0 (boundary): re-mask probability strictly in (0, 1)
    // and seed actually affects the output.
    let mut distinct_outputs = std::collections::HashSet::new();
    for seed in 1u64..=16 {
        let out = oracle.step(&x_t, &mask, &clean, 0, seed);
        distinct_outputs.insert(out);
    }
    assert!(
        distinct_outputs.len() >= 4,
        "LCG seed must produce distinct re-mask tensors for distinct seeds; \
         got {} distinct outputs across seeds 1..=16 (suspected seed is ignored)",
        distinct_outputs.len()
    );
}
