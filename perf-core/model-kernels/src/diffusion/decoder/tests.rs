//! Oracle tests (TDD) for [`super::DiffusionDecoder`].
//!
//! See the parent module docs for the LLaDA / Dream acceptance trace
//! algorithm.

use crate::common::Lcg;
use crate::error::KernelError;

use super::super::denoise::RemaskStrategy;
use super::decoder::DiffusionDecoder;

/// Build a logits row that yields softmax-max = 0.5 for `vocab=8`.
/// logit x at the argmax index with e^x / (e^x + (vocab - 1)) = 0.5
/// → e^x = (vocab - 1) → x = ln(vocab - 1).
fn uniform_half_confidence_logits(n: usize, vocab: usize, argmax: usize) -> Vec<f32> {
    assert!(vocab > 1);
    assert!(argmax < vocab);
    let x = ((vocab - 1) as f64).ln() as f32;
    let mut out = Vec::with_capacity(n * vocab);
    for _ in 0..n {
        for j in 0..vocab {
            if j == argmax {
                out.push(x);
            } else {
                out.push(0.0);
            }
        }
    }
    out
}

#[test]
fn new_rejects_zero_vocab() {
    let err = DiffusionDecoder::new(
        0,
        0,
        4,
        RemaskStrategy::LowConfidence { percentile: 50.0 },
        0.0,
    )
    .unwrap_err();
    assert!(matches!(err, KernelError::ZeroDimension { .. }));
}

#[test]
fn new_rejects_non_power_of_two_vocab() {
    let err = DiffusionDecoder::new(
        6, // not a power of two
        0,
        4,
        RemaskStrategy::LowConfidence { percentile: 50.0 },
        0.0,
    )
    .unwrap_err();
    assert!(matches!(err, KernelError::BadBufferLength { .. }));
}

#[test]
fn new_rejects_mask_token_outside_vocab() {
    let err = DiffusionDecoder::new(
        8,
        8, // >= vocab
        4,
        RemaskStrategy::LowConfidence { percentile: 50.0 },
        0.0,
    )
    .unwrap_err();
    assert!(matches!(err, KernelError::OutOfRange { .. }));
}

#[test]
fn new_rejects_zero_total_steps() {
    let err = DiffusionDecoder::new(
        8,
        0,
        0,
        RemaskStrategy::LowConfidence { percentile: 50.0 },
        0.0,
    )
    .unwrap_err();
    assert!(matches!(err, KernelError::BadBufferLength { .. }));
}

#[test]
fn new_rejects_invalid_confidence_threshold() {
    let err = DiffusionDecoder::new(
        8,
        0,
        4,
        RemaskStrategy::LowConfidence { percentile: 50.0 },
        1.5,
    )
    .unwrap_err();
    assert!(matches!(err, KernelError::OutOfRange { .. }));
}

#[test]
fn step_returns_error_on_mask_length_mismatch() {
    let dec = DiffusionDecoder::new(
        8,
        0,
        4,
        RemaskStrategy::LowConfidence { percentile: 50.0 },
        0.0,
    )
    .unwrap();
    let mut x_t = vec![0u32; 4];
    let mut mask = vec![true, true, true]; // mismatched
    let logits = vec![0.0f32; 4 * 8];
    let err = dec.step(&mut x_t, &mut mask, &logits).unwrap_err();
    assert!(matches!(err, KernelError::DimMismatch { .. }));
}

#[test]
fn step_returns_error_on_logits_length_mismatch() {
    let dec = DiffusionDecoder::new(
        8,
        0,
        4,
        RemaskStrategy::LowConfidence { percentile: 50.0 },
        0.0,
    )
    .unwrap();
    let mut x_t = vec![0u32; 4];
    let mut mask = vec![true; 4];
    let logits = vec![0.0f32; 3 * 8]; // wrong length
    let err = dec.step(&mut x_t, &mut mask, &logits).unwrap_err();
    assert!(matches!(err, KernelError::BadBufferLength { .. }));
}

#[test]
fn oracle_low_confidence_finishes_with_unmasked_tokens() {
    // 32-token sequence, vocab=8, mask_token=0, 4 denoise steps with
    // LowConfidence { percentile: 50.0 }. The trace must finish
    // with every position unmasked AND every mask_token-initial
    // position must have a non-mask_token value.
    //
    // Construction: logits rows with softmax-max = 0.5 uniformly
    // across all 32 positions. Under the 50th-percentile remask,
    // threshold = median = 0.5 and `score < 0.5` is false for all,
    // so nothing gets re-masked. Every step is a clean decode.
    let n = 32usize;
    let vocab = 8usize;
    let mask_token = 0u32;
    let argmax_token = 7usize; // non-mask_token within vocab
    let dec = DiffusionDecoder::new(
        vocab,
        mask_token,
        4,
        RemaskStrategy::LowConfidence { percentile: 50.0 },
        0.0,
    )
    .unwrap();
    let mut x_t = vec![mask_token; n];
    let mut mask = vec![true; n];
    let logits = uniform_half_confidence_logits(n, vocab, argmax_token);
    for _step in 0..4 {
        dec.step(&mut x_t, &mut mask, &logits).unwrap();
    }
    assert!(
        mask.iter().all(|m| !*m),
        "every position must be unmasked after 4 steps"
    );
    assert!(
        x_t.iter().all(|&t| t != mask_token),
        "every mask_token-initial position must have a non-mask_token value"
    );
    assert!(x_t.iter().all(|&t| t == argmax_token as u32));
}

#[test]
fn oracle_deterministic_replay_produces_identical_sequences() {
    // Two runs with the same seed must produce identical final
    // sequences. Use an LCG-driven logits generator so determinism
    // is observable (constant logits would trivially satisfy the
    // assertion).
    let n = 32usize;
    let vocab = 8usize;
    let mask_token = 0u32;
    let total_steps = 4usize;
    let dec = DiffusionDecoder::new(
        vocab,
        mask_token,
        total_steps,
        RemaskStrategy::LowConfidence { percentile: 50.0 },
        0.0,
    )
    .unwrap();

    let mut rng1 = Lcg::new(0xDEAD_BEEF);
    let mut rng2 = Lcg::new(0xDEAD_BEEF);
    let mut x1 = vec![mask_token; n];
    let mut x2 = vec![mask_token; n];
    let mut m1 = vec![true; n];
    let mut m2 = vec![true; n];
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

#[test]
fn oracle_remask_strategy_validation_invalid_percentile_returns_out_of_range() {
    // Construct a decoder with an invalid percentile and confirm
    // the error propagates out of step() as OutOfRange.
    let dec_bad = DiffusionDecoder::new(
        8,
        0,
        4,
        RemaskStrategy::LowConfidence { percentile: -10.0 },
        0.0,
    )
    .unwrap();
    let mut x_t = vec![0u32; 4];
    let mut mask = vec![true; 4];
    let logits = vec![0.0f32; 4 * 8];
    let err = dec_bad.step(&mut x_t, &mut mask, &logits).unwrap_err();
    assert!(matches!(err, KernelError::OutOfRange { .. }));
}
