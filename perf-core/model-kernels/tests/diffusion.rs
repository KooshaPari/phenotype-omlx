//! LLaDA / Dream diffusion acceptance integration tests.
//!
//! Drives the [`model_kernels::diffusion::DiffusionDecoder`] orchestrator
//! end-to-end against the two diffusion rows of the model acceptance
//! matrix in `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md`:
//!
//! - LLaDA — `LowConfidence { percentile: 60.0 }` with
//!   `confidence_threshold = 0.5`, mask token = 15, vocab = 16,
//!   64-token sequence, 8 steps.
//! - Dream — `EntropyBased` strategy with the same shape.
//!
//! These tests are acceptance traces, not unit tests of `denoise_step`.
//! The denoise / remask kernels themselves are exercised by
//! `tests/contracts.rs`.

use model_kernels::common::Lcg;
use model_kernels::diffusion::{DiffusionDecoder, DiffusionStepReport, RemaskStrategy};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

const SEED: u64 = 0xCAFE_BABE;

/// Build logits with a *step-dependent* monotonically-growing
/// high-confidence prefix:
/// at step `t`, positions `[0..prefix_size(t))` have the high
/// confidence logit; the rest have the low confidence logit.
///
/// The schedule for LLaDA is `prefix_size(0)=32`, growing by `16`
/// each step, capped at `n`. So:
///   step 0 → [0..32] high,   step 1 → [0..48] high,
///   step 2 → [0..64] high,   step 3..7 → [0..64] high.
/// Every step's high set is a superset of the previous, so the
/// unmasked count is monotonically non-decreasing.
fn staged_confidence_logits(
    n: usize,
    vocab: usize,
    step: usize,
    high_logit: f32,
    low_logit: f32,
    prefix_step0: usize,
    prefix_step_increment: usize,
) -> Vec<f32> {
    assert!(vocab >= 1);
    let target_prefix = (prefix_step0 + prefix_step_increment * step).min(n);
    let mut out = Vec::with_capacity(n * vocab);
    for i in 0..n {
        let logit = if i < target_prefix { high_logit } else { low_logit };
        for j in 0..vocab {
            out.push(if j == 0 { logit } else { 0.0 });
        }
    }
    out
}

/// Drive a diffusion trace on a fresh `x_t`/`mask` using a step-index
/// logit generator and return the per-step reports together with the
/// final `x_t` and `mask`.
fn run_trace_with_logit_gen<F>(
    decoder: &DiffusionDecoder,
    n: usize,
    total_steps: usize,
    mut logit_for_step: F,
) -> (Vec<u32>, Vec<bool>, Vec<DiffusionStepReport>)
where
    F: FnMut(usize) -> Vec<f32>,
{
    let mut x_t = vec![decoder.mask_token(); n];
    let mut mask = vec![true; n];
    let mut reports = Vec::with_capacity(total_steps);
    for step in 0..total_steps {
        let logits = logit_for_step(step);
        let mut report = decoder.step(&mut x_t, &mut mask, &logits).unwrap();
        report.step = step;
        reports.push(report);
    }
    (x_t, mask, reports)
}

/// Number of positions in `mask` that are currently unmasked.
fn count_unmasked(mask: &[bool]) -> usize {
    mask.iter().filter(|m| !**m).count()
}

// ---------------------------------------------------------------------------
// Logit constants for the staged LLaDA + Dream traces
// ---------------------------------------------------------------------------

// vocab = 16:
// softmax-max = e^x / (e^x + 15) = 0.3  → e^x = 6.4286 → x = ln(6.4286) ≈ 1.861
// softmax-max = e^x / (e^x + 15) = 0.7  → e^x = 35       → x = ln(35)    ≈ 3.555
const LOW_CONF_LOGIT_V16: f32 = 1.860_752_f32; // ln(6.4286) cast
const HIGH_CONF_LOGIT_V16: f32 = 3.555_348_f32; // ln(35.0) cast

// ---------------------------------------------------------------------------
// LLaDA acceptance trace (vocab=16, mask_token=15, total_steps=8,
// LowConfidence { percentile: 60.0 }, confidence_threshold=0.5)
// ---------------------------------------------------------------------------

/// (a) After running the full LLaDA trace every position must be
/// unmasked and no position should retain the mask token.
///
/// Logits schedule: high-confidence prefix grows 32 → 48 → 64 across
/// the first three steps; positions in the prefix have score 0.7,
/// positions outside have score 0.3. With confidence_threshold=0.5
/// the low positions are force-remasked, and with percentile 60 the
/// threshold = 0.7 (since high-conf scores dominate the 60th
/// percentile), so `score < 0.7` re-masks the low positions again.
/// The prefix never shrinks, so unmasked-count is monotonic.
#[test]
fn llada_acceptance_trace_finishes_with_all_unmasked() {
    let n = 64usize;
    let vocab = 16usize;
    let mask_token = 15u32;
    let total_steps = 8usize;
    let dec = DiffusionDecoder::new(
        vocab,
        mask_token,
        total_steps,
        RemaskStrategy::LowConfidence { percentile: 60.0 },
        0.5,
    )
    .unwrap();

    let (x_t, mask, reports) = run_trace_with_logit_gen(&dec, n, total_steps, |step| {
        staged_confidence_logits(
            n,
            vocab,
            step,
            HIGH_CONF_LOGIT_V16,
            LOW_CONF_LOGIT_V16,
            32, // prefix_step0
            16, // prefix_step_increment
        )
    });

    assert!(
        mask.iter().all(|m| !*m),
        "every position must be unmasked after {total_steps} steps (mask={mask:?})"
    );
    assert!(reports.last().unwrap().finished);
    assert!(x_t.iter().all(|&t| t != mask_token));
}

/// (b) Two runs of the LLaDA trace with the same seed must produce
/// identical final `x_t` and `mask` vectors.
#[test]
fn llada_acceptance_trace_is_deterministic_across_runs() {
    let n = 64usize;
    let vocab = 16usize;
    let mask_token = 15u32;
    let total_steps = 8usize;
    let dec = DiffusionDecoder::new(
        vocab,
        mask_token,
        total_steps,
        RemaskStrategy::LowConfidence { percentile: 60.0 },
        0.5,
    )
    .unwrap();

    let mut x1 = vec![mask_token; n];
    let mut m1 = vec![true; n];
    let mut x2 = vec![mask_token; n];
    let mut m2 = vec![true; n];
    let mut rng1 = Lcg::new(SEED ^ 0x11ADA);
    let mut rng2 = Lcg::new(SEED ^ 0x11ADA);
    for _ in 0..total_steps {
        let logits1: Vec<f32> = (0..n * vocab).map(|_| rng1.next_signed()).collect();
        let logits2: Vec<f32> = (0..n * vocab).map(|_| rng2.next_signed()).collect();
        assert_eq!(logits1, logits2, "LCG with same seed must match");
        dec.step(&mut x1, &mut m1, &logits1).unwrap();
        dec.step(&mut x2, &mut m2, &logits2).unwrap();
    }
    assert_eq!(x1, x2);
    assert_eq!(m1, m2);
}

/// (c) The running count of unmasked positions must be monotonically
/// non-decreasing across the trace. Once a position is unmasked it
/// must stay unmasked; this is the contract that distinguishes a
/// converging diffusion trace from one that thrashes.
#[test]
fn llada_running_unmasked_count_is_monotonically_non_decreasing() {
    let n = 64usize;
    let vocab = 16usize;
    let mask_token = 15u32;
    let total_steps = 8usize;
    let dec = DiffusionDecoder::new(
        vocab,
        mask_token,
        total_steps,
        RemaskStrategy::LowConfidence { percentile: 60.0 },
        0.5,
    )
    .unwrap();
    let mut x_t = vec![mask_token; n];
    let mut mask = vec![true; n];

    let mut prev_unmasked = 0usize;
    for step in 0..total_steps {
        let logits = staged_confidence_logits(
            n,
            vocab,
            step,
            HIGH_CONF_LOGIT_V16,
            LOW_CONF_LOGIT_V16,
            32,
            16,
        );
        dec.step(&mut x_t, &mut mask, &logits).unwrap();
        let unmasked = count_unmasked(&mask);
        assert!(
            unmasked >= prev_unmasked,
            "step {step}: unmasked count must be monotonically non-decreasing \
             (prev={prev_unmasked}, now={unmasked})"
        );
        prev_unmasked = unmasked;
    }
    // Sanity: every position unmasked by the end.
    assert_eq!(prev_unmasked, n);
}

/// (d) Across the full trace, no position may be re-masked more than
/// `total_steps - 1` times.
#[test]
fn llada_remask_count_per_position_bounded_by_total_steps_minus_one() {
    let n = 64usize;
    let vocab = 16usize;
    let mask_token = 15u32;
    let total_steps = 8usize;
    let dec = DiffusionDecoder::new(
        vocab,
        mask_token,
        total_steps,
        RemaskStrategy::LowConfidence { percentile: 60.0 },
        0.5,
    )
    .unwrap();
    let mut x_t = vec![mask_token; n];
    let mut mask = vec![true; n];

    let mut per_position_remasks = vec![0usize; n];
    let mut prev_mask: Vec<bool> = mask.clone();
    for step in 0..total_steps {
        let logits = staged_confidence_logits(
            n,
            vocab,
            step,
            HIGH_CONF_LOGIT_V16,
            LOW_CONF_LOGIT_V16,
            32,
            16,
        );
        dec.step(&mut x_t, &mut mask, &logits).unwrap();
        for i in 0..n {
            if mask[i] && !prev_mask[i] {
                per_position_remasks[i] += 1;
            }
        }
        prev_mask = mask.clone();
    }
    for (i, &c) in per_position_remasks.iter().enumerate() {
        assert!(
            c < total_steps,
            "position {i} remasked {c} times, exceeds total_steps = {total_steps}"
        );
    }
}

// ---------------------------------------------------------------------------
// Dream acceptance trace (same shape, EntropyBased strategy)
// ---------------------------------------------------------------------------

// For the Dream-style trace we use a smaller low-confidence logit so
// the EntropyBased 25th-percentile threshold (= sorted[16]) actually
// pulls in the low-confidence positions. score = 0.2 → e^x = 3.0 → x = ln(3).
const DREAM_LOW_CONF_LOGIT_V16: f32 = 1.098_612_3_f32; // ln(3) cast

/// (a) Dream-style trace: after 8 steps every position must be unmasked
/// and no position retains the mask token.
#[test]
fn dream_acceptance_trace_finishes_with_all_unmasked() {
    let n = 64usize;
    let vocab = 16usize;
    let mask_token = 15u32;
    let total_steps = 8usize;
    let dec = DiffusionDecoder::new(
        vocab,
        mask_token,
        total_steps,
        RemaskStrategy::EntropyBased,
        0.0, // no confidence floor
    )
    .unwrap();

    // Dream grows the prefix faster: 48 → 56 → 64.
    let (x_t, mask, reports) = run_trace_with_logit_gen(&dec, n, total_steps, |step| {
        staged_confidence_logits(
            n,
            vocab,
            step,
            HIGH_CONF_LOGIT_V16,
            DREAM_LOW_CONF_LOGIT_V16,
            48, // prefix_step0
            8,  // prefix_step_increment
        )
    });

    assert!(
        mask.iter().all(|m| !*m),
        "every position must be unmasked after {total_steps} steps (mask={mask:?})"
    );
    assert!(reports.last().unwrap().finished);
    assert!(x_t.iter().all(|&t| t != mask_token));
}

/// (b) Dream-style determinism: same seed → identical final state.
#[test]
fn dream_acceptance_trace_is_deterministic_across_runs() {
    let n = 64usize;
    let vocab = 16usize;
    let mask_token = 15u32;
    let total_steps = 8usize;
    let dec = DiffusionDecoder::new(
        vocab,
        mask_token,
        total_steps,
        RemaskStrategy::EntropyBased,
        0.0,
    )
    .unwrap();

    let mut x1 = vec![mask_token; n];
    let mut m1 = vec![true; n];
    let mut x2 = vec![mask_token; n];
    let mut m2 = vec![true; n];
    let mut rng1 = Lcg::new(SEED ^ 0xD2EAA);
    let mut rng2 = Lcg::new(SEED ^ 0xD2EAA);
    for _ in 0..total_steps {
        let logits1: Vec<f32> = (0..n * vocab).map(|_| rng1.next_signed()).collect();
        let logits2: Vec<f32> = (0..n * vocab).map(|_| rng2.next_signed()).collect();
        assert_eq!(logits1, logits2, "LCG with same seed must match");
        dec.step(&mut x1, &mut m1, &logits1).unwrap();
        dec.step(&mut x2, &mut m2, &logits2).unwrap();
    }
    assert_eq!(x1, x2);
    assert_eq!(m1, m2);
}

/// (c) Dream-style: unmasked count is monotonically non-decreasing.
#[test]
fn dream_running_unmasked_count_is_monotonically_non_decreasing() {
    let n = 64usize;
    let vocab = 16usize;
    let mask_token = 15u32;
    let total_steps = 8usize;
    let dec = DiffusionDecoder::new(
        vocab,
        mask_token,
        total_steps,
        RemaskStrategy::EntropyBased,
        0.0,
    )
    .unwrap();
    let mut x_t = vec![mask_token; n];
    let mut mask = vec![true; n];

    let mut prev_unmasked = 0usize;
    for step in 0..total_steps {
        let logits = staged_confidence_logits(
            n,
            vocab,
            step,
            HIGH_CONF_LOGIT_V16,
            DREAM_LOW_CONF_LOGIT_V16,
            48,
            8,
        );
        dec.step(&mut x_t, &mut mask, &logits).unwrap();
        let unmasked = count_unmasked(&mask);
        assert!(
            unmasked >= prev_unmasked,
            "step {step}: unmasked count must be monotonically non-decreasing \
             (prev={prev_unmasked}, now={unmasked})"
        );
        prev_unmasked = unmasked;
    }
    assert_eq!(prev_unmasked, n);
}

/// (d) Dream-style: per-position remask count bounded by total_steps - 1.
#[test]
fn dream_remask_count_per_position_bounded_by_total_steps_minus_one() {
    let n = 64usize;
    let vocab = 16usize;
    let mask_token = 15u32;
    let total_steps = 8usize;
    let dec = DiffusionDecoder::new(
        vocab,
        mask_token,
        total_steps,
        RemaskStrategy::EntropyBased,
        0.0,
    )
    .unwrap();
    let mut x_t = vec![mask_token; n];
    let mut mask = vec![true; n];

    let mut per_position_remasks = vec![0usize; n];
    let mut prev_mask: Vec<bool> = mask.clone();
    for step in 0..total_steps {
        let logits = staged_confidence_logits(
            n,
            vocab,
            step,
            HIGH_CONF_LOGIT_V16,
            DREAM_LOW_CONF_LOGIT_V16,
            48,
            8,
        );
        dec.step(&mut x_t, &mut mask, &logits).unwrap();
        for i in 0..n {
            if mask[i] && !prev_mask[i] {
                per_position_remasks[i] += 1;
            }
        }
        prev_mask = mask.clone();
    }
    for (i, &c) in per_position_remasks.iter().enumerate() {
        assert!(
            c < total_steps,
            "position {i} remasked {c} times, exceeds total_steps = {total_steps}"
        );
    }
}