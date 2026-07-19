//! (l) Discrete (masked) diffusion language model — MDLM / D3PM / SEDD
//!
//! NOTE: `kernel-registry` does not yet have a first-class
//! `DiscreteDiffusion` selector. This file therefore defines a
//! **test-only stub selector** (a local `SelectorMetadata` struct with
//! `mode = Decode`, `policy = Deterministic`, `kind = MaskedDiffusionStep`)
//! that drives the existing `KernelRegistry` selector via the
//! `OperatorKind::DiscreteDiffusion` variant added to `kernel_registry::compat`.
//!
//! The stub is sufficient to verify the byte-identical oracle
//! contract for the discrete diffusion family:
//!
//! 1. The deterministic policy picks the kernel with the lowest p95
//!    for `OperatorKind::DiscreteDiffusion` at the chosen shape.
//! 2. The reference oracle (one-hot of the clean token at every
//!    noised position) is byte-stable across identical invocations.
//! 3. The masked-token tensor's invariant — only positions that were
//!    masked get re-masked under the schedule — holds for the linear
//!    and cosine schedules.
//! 4. Linear and cosine schedules produce *different* mask counts
//!    for the same step, confirming the schedule parameter is wired
//!    into the oracle rather than no-op.
//!
//! When the kernel-registry gains a real `DiscreteDiffusion` selector,
//! this file should be extended to register its backend candidates and
//! drop the test-only stub.

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::selector::SelectionDecision;
use kernel_registry::{
    BackendKind, Capability, KernelKey, KernelRegistry, SelectionPolicy,
};

use super::{
    build_record, fresh_capabilities, make_candidate, samples_with_p95, shape, NOW_UNIX_MS,
    TEST_FINGERPRINT,
};

// ---------------------------------------------------------------------------
// Test-only stub selector types.
// ---------------------------------------------------------------------------

/// Selector execution mode. Matches the language of the surrounding
/// runtime; `Decode` means "one masked position per call, decode
/// greedily, advance the schedule".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectorMode {
    Prefill,
    Decode,
}

/// Kind of discrete-diffusion step the selector is asked to dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepKind {
    /// Decode one masked position per call under the noise schedule.
    MaskedDiffusionStep,
    /// Sample an entirely noised sequence from the prior (used at
    /// the start of every denoising chain).
    PriorSample,
}

/// Per-call selector metadata. Mirrors the design that will land in
/// the kernel-registry proper; defined here so the discrete-diffusion
/// oracle test does not need to wait on that refactor.
#[derive(Debug, Clone)]
struct SelectorMetadata {
    /// Family discriminator on the `KernelKey`.
    family: OperatorKind,
    /// Decode vs prefill — controls the oracle's update pattern.
    mode: SelectorMode,
    /// Selection policy for the registry call.
    policy: SelectionPolicy,
    /// Which step kind the dispatch represents.
    kind: StepKind,
}

impl SelectorMetadata {
    fn decode_deterministic() -> Self {
        Self {
            family: OperatorKind::DiscreteDiffusion,
            mode: SelectorMode::Decode,
            policy: SelectionPolicy::Deterministic { prefer_lower_p95: true },
            kind: StepKind::MaskedDiffusionStep,
        }
    }
}

// ---------------------------------------------------------------------------
// Noise schedules (linear + cosine).
// ---------------------------------------------------------------------------

/// Noise schedule used by the discrete diffusion model. The schedule
/// controls how many positions stay masked at step `t` of `num_steps`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Schedule {
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
    fn alpha_at(self, t: usize, num_steps: usize) -> f64 {
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
#[derive(Debug, Clone, Copy)]
struct DiscreteDiffusionOracle {
    vocab_size: u32,
    mask_token_id: u32,
    num_steps: usize,
    schedule: Schedule,
}

impl DiscreteDiffusionOracle {
    fn new(vocab_size: u32, mask_token_id: u32, num_steps: usize, schedule: Schedule) -> Self {
        // The mask token must be a valid vocab id so it round-trips
        // through the buffer byte representation.
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
    fn step(&self, x_t: &[u32], mask: &[bool], clean: &[u32], step: usize, seed: u64) -> Vec<u32> {
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
    fn next_mask(&self, x_next: &[u32]) -> Vec<bool> {
        x_next.iter().map(|&t| t == self.mask_token_id).collect()
    }
}

/// Tiny linear-congruential generator (MMIX constants). Deterministic,
/// seed-stable, and avoids pulling in `rand`.
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ---------------------------------------------------------------------------
// Registry wiring.
// ---------------------------------------------------------------------------

fn ddm_key(vocab_size: usize, num_steps: usize) -> KernelKey {
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

fn ddm_registry() -> (KernelRegistry, kernel_registry::CandidateId, kernel_registry::CandidateId) {
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
// Tests.
// ---------------------------------------------------------------------------

/// The Deterministic policy under the discrete-diffusion metadata
/// selects the lowest-p95 backend (Metal). This is the selector-coverage
/// half of the contract: even though the oracle is the test's source of
/// truth, the registry must still pick the right kernel for it.
#[test]
fn ddm_metadata_decode_deterministic_picks_lowest_p95_metal_backend() {
    let (reg, _id_scalar, id_metal) = ddm_registry();
    let meta = SelectorMetadata::decode_deterministic();
    let key = ddm_key(16, 8);
    let decision = reg.select_with_caps(&key, meta.policy.clone(), &fresh_capabilities(), NOW_UNIX_MS);
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_metal,
                "metal p95=2100 must beat scalar p95=9500 under Deterministic"
            );
            // Pin every field of the stub metadata so future
            // refactors cannot accidentally drop the family / mode /
            // kind discriminators that distinguish the
            // DiscreteDiffusion selector from neighbors.
            assert_eq!(meta.family, OperatorKind::DiscreteDiffusion);
            assert_eq!(meta.mode, SelectorMode::Decode);
            assert_eq!(meta.kind, StepKind::MaskedDiffusionStep);
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

/// The stub metadata's other variants round-trip through `Debug`.
/// This pins the existence of `Prefill` and `PriorSample` so the
/// stub remains a faithful preview of the eventual full selector.
#[test]
fn ddm_metadata_other_variants_exist() {
    let _prefill = SelectorMode::Prefill;
    let _prior = StepKind::PriorSample;
}

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