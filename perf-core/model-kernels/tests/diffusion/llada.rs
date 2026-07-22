//! LLaDA acceptance trace (vocab=16, mask_token=15, total_steps=8,
//! LowConfidence { percentile: 60.0 }, confidence_threshold=0.5).

use super::*;

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
