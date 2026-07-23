//! Section "Diffusion" of the original contracts.rs.
//!
//! Split out of the original monolithic `model-kernels/tests/contracts.rs`
//! (1130 lines) so each topic stays under the 350-line target. Test bodies
//! are byte-identical to the source file; only the surrounding module
//! wrapper and `use super::*;` import differ.

use super::*;

#[test]
fn denoise_step_updates_only_masked_tokens() {
    // Two tokens, the second is masked.
    let vocab = 4;
    let x_t = vec![0u32, 0];
    let mask = vec![false, true];
    // Model logits: at position 1 we predict token 3 with high confidence;
    // at position 0 we predict token 2 (we'll never see it because it's
    // already unmasked).
    let model_logits = vec![
        // position 0
        0.0, 0.1, 5.0, 0.0, // position 1
        0.0, 0.1, 0.2, 9.0,
    ];
    let upd: DenoiseUpdate =
        denoise_step(&x_t, &mask, &model_logits, RemaskStrategy::None, 0.0, vocab).unwrap();
    // Position 0 untouched.
    assert_eq!(upd.next_x[0], 0);
    assert!(!upd.next_mask[0]);
    // Position 1 was masked and remask=None, so it accepts its argmax.
    assert_eq!(upd.next_x[1], 3);
    assert!(!upd.next_mask[1]);
    assert_eq!(upd.accepted_count, 1);
}

#[test]
fn denoise_step_with_no_remask_strategy_leaves_mask_unchanged() {
    // Two tokens, both masked. With RemaskStrategy::None both should
    // accept their argmax and no positions should remain masked.
    let vocab = 3;
    let x_t = vec![0u32, 0];
    let mask = vec![true, true];
    let model_logits = vec![0.0, 5.0, 0.1, 2.0, 0.0, 0.0];
    let upd = denoise_step(&x_t, &mask, &model_logits, RemaskStrategy::None, 0.0, vocab).unwrap();
    for &m in &upd.next_mask {
        assert!(!m);
    }
    assert_eq!(upd.accepted_count, 2);
}

#[test]
fn remask_low_confidence_respects_percentile() {
    // 4 tokens; set confidences [0.9, 0.8, 0.2, 0.1]. Percentile=50
    // means the *lower* half (relative to confidence) is re-masked.
    let scores = vec![0.9, 0.8, 0.2, 0.1];
    let mut mask = vec![true, true, true, true];
    remask(
        &scores,
        &mut mask,
        &RemaskStrategy::LowConfidence { percentile: 50.0 },
        0,
        1,
    )
    .unwrap();
    // Tokens 0 and 1 (confidence >= median) are accepted; tokens 2 and 3
    // are re-masked.
    assert!(!mask[0]);
    assert!(!mask[1]);
    assert!(mask[2]);
    assert!(mask[3]);
}

#[test]
fn confidence_scores_match_softmax_max() {
    let logits = vec![1.0, 2.0, 5.0, 0.0];
    let scores = confidence_scores(&logits, 4).unwrap();
    // Softmax max should equal exp(5) / sum(exp(.)) = exp(5)/Z.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: f32 = logits.iter().map(|l| (l - max).exp()).sum();
    let expected = (5.0f32 - max).exp() / exp;
    assert!((scores[0] - expected).abs() < 1e-5);
    assert_eq!(scores.len(), 1);
}

#[test]
fn parallel_denoise_matches_sequential_denoise() {
    let vocab = 4;
    let n = 6;
    let mut rng = Lcg::new(SEED ^ 0xD1FF);
    let x_t: Vec<u32> = (0..n).map(|i| (i as u32) % vocab as u32).collect();
    let mask: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let model_logits: Vec<f32> = (0..n * vocab).map(|_| rng.next_signed()).collect();

    let upd = denoise_step(
        &x_t,
        &mask,
        &model_logits,
        RemaskStrategy::RandomFraction(0.0),
        0.0,
        vocab,
    )
    .unwrap();
    // With remask fraction 0, no tokens get re-masked: any masked token
    // accepts its argmax.
    for i in 0..n {
        if mask[i] {
            assert!(
                !upd.next_mask[i],
                "masked token {i} should accept at frac=0"
            );
        } else {
            assert_eq!(upd.next_x[i], x_t[i]);
        }
    }
}
