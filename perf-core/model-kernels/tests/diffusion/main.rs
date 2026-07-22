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
//!
//! The suite is split across per-topic sub-modules:
//!
//! - [`llada`] — `LowConfidence { percentile: 60.0 }` acceptance trace.
//! - [`dream`] — `EntropyBased` strategy acceptance trace.

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

// For the Dream-style trace we use a smaller low-confidence logit so
// the EntropyBased 25th-percentile threshold (= sorted[16]) actually
// pulls in the low-confidence positions. score = 0.2 → e^x = 3.0 → x = ln(3).
const DREAM_LOW_CONF_LOGIT_V16: f32 = 1.098_612_3_f32; // ln(3) cast

mod dream;
mod llada;
