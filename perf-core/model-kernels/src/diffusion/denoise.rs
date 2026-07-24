//! Denoise step kernels for diffusion language models.
//!
//! The pure remask scheduler lives in [`super::remask`]; this module
//! only owns [`denoise_step_sequential`] (the per-row argmax decode)
//! and the fused [`denoise_step`] that calls it and then applies a
//! remask strategy.

use crate::common::softmax_max;
use crate::error::{KernelError, Result};

use super::remask::remask;

/// Strategy for choosing which newly-unmasked tokens to re-mask
/// during a denoise step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RemaskStrategy {
    /// Re-mask tokens whose confidence falls below a percentile of
    /// the confidence distribution for this step. `percentile` is
    /// expressed on a `0..=100` scale (50 = median).
    LowConfidence { percentile: f32 },
    /// Re-mask tokens with the highest Shannon entropy in the
    /// softmax distribution (least decisive). The current
    /// implementation uses a 25th-percentile surrogate for
    /// determinism.
    EntropyBased,
    /// Re-mask a uniformly random fraction of newly-unmasked tokens.
    /// Seeded deterministically from `(step, total_steps)`.
    RandomFraction(f32),
    /// Do not re-mask anything; once a token is decoded, it stays.
    None,
}

/// Result of one denoise step.
#[derive(Debug, Clone, PartialEq)]
pub struct DenoiseUpdate {
    /// Next-token IDs after the step.
    pub next_x: Vec<u32>,
    /// Next mask after the step (true = still masked).
    pub next_mask: Vec<bool>,
    /// Number of positions that are *not* masked after the step.
    pub accepted_count: usize,
}

/// Sequential scalar reference used for DiffusionGemma parity and
/// retained for LLaDA/Dream regression fixtures. One row at a time, picks the argmax
/// for any position currently in `mask`, fills it into `next_x`,
/// leaves other positions alone.
pub fn denoise_step_sequential(
    x_t: &[u32],
    mask: &[bool],
    model_logits: &[f32],
    vocab: usize,
) -> Result<DenoiseUpdate> {
    let n = x_t.len();
    if mask.len() != n {
        return Err(KernelError::DimMismatch {
            what: "denoise_step.mask vs x_t",
            expected: n,
            got: mask.len(),
        });
    }
    if model_logits.len() != n * vocab {
        return Err(KernelError::BadBufferLength {
            what: "model_logits",
            expected: n * vocab,
            got: model_logits.len(),
        });
    }
    let mut next_x = x_t.to_vec();
    let mut next_mask = mask.to_vec();
    let mut accepted = 0usize;
    for i in 0..n {
        if !mask[i] {
            continue;
        }
        let row = &model_logits[i * vocab..(i + 1) * vocab];
        let (best_id, _) = argmax(row);
        next_x[i] = best_id as u32;
        next_mask[i] = false;
        accepted += 1;
    }
    Ok(DenoiseUpdate {
        next_x,
        next_mask,
        accepted_count: accepted,
    })
}

/// Parallel fused variant: decodes masked positions, then applies the
/// remask strategy using per-position softmax-max scores derived from
/// the supplied logits.
///
/// `confidence_threshold` is an absolute floor; anything below the
/// floor is forced back to masked even if `strategy` would have left
/// it alone.
#[allow(clippy::too_many_arguments)]
pub fn denoise_step(
    x_t: &[u32],
    mask: &[bool],
    model_logits: &[f32],
    strategy: RemaskStrategy,
    confidence_threshold: f32,
    vocab: usize,
) -> Result<DenoiseUpdate> {
    // Count positions that were originally masked (= newly accepted
    // if no remask is applied).
    let newly_accepted = mask.iter().filter(|m| **m).count();
    // 1. Sequential decode for masked positions.
    let mut update = denoise_step_sequential(x_t, mask, model_logits, vocab)?;
    // 2. Compute per-position softmax-max confidence scores.
    let n = update.next_mask.len();
    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let row = &model_logits[i * vocab..(i + 1) * vocab];
        scores.push(softmax_max(row));
    }
    // 3. Confidence threshold: anything below is forced back to masked.
    if !(0.0..=1.0).contains(&confidence_threshold) {
        return Err(KernelError::OutOfRange {
            what: "confidence_threshold",
            min: 0.0,
            max: 1.0,
            got: confidence_threshold,
        });
    }
    if confidence_threshold > 0.0 {
        for (i, &s) in scores.iter().enumerate() {
            if s < confidence_threshold {
                update.next_mask[i] = true;
            }
        }
    }
    // 4. Apply remask strategy.
    remask(&scores, &mut update.next_mask, &strategy, 0, 1)?;
    // 5. accepted_count = originally-masked - re-masked.
    let re_masked_count = update
        .next_mask
        .iter()
        .zip(mask.iter())
        .filter(|(m_new, m_old)| **m_new && !**m_old)
        .count();
    update.accepted_count = newly_accepted.saturating_sub(re_masked_count);
    Ok(update)
}

fn argmax(row: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    (best, best_v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_only_updates_masked_positions() {
        let x_t = vec![0u32, 0, 0, 0];
        let mask = vec![true, false, true, false];
        let logits = vec![
            0.0, 0.0, 10.0, 0.0, // -> 2
            0.0, 0.0, 0.0, 0.0, // unused
            0.0, 5.0, 0.0, 0.0, // -> 1
            0.0, 0.0, 0.0, 0.0, // unused
        ];
        let out = denoise_step_sequential(&x_t, &mask, &logits, 4).unwrap();
        assert_eq!(out.next_x, vec![2, 0, 1, 0]);
        assert_eq!(out.next_mask, vec![false, false, false, false]);
        assert_eq!(out.accepted_count, 2);
    }

    #[test]
    fn rejects_mask_length_mismatch() {
        let x_t = vec![0u32, 0];
        let mask = vec![true];
        let logits = vec![0.0, 0.0];
        let err = denoise_step_sequential(&x_t, &mask, &logits, 2).unwrap_err();
        assert!(matches!(err, KernelError::DimMismatch { .. }));
    }

    #[test]
    fn rejects_logit_length_mismatch() {
        let x_t = vec![0u32, 0];
        let mask = vec![true, true];
        let logits = vec![0.0; 3];
        let err = denoise_step_sequential(&x_t, &mask, &logits, 2).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn parallel_matches_sequential_when_no_remask() {
        let x_t = vec![3u32, 1, 2];
        let mask = vec![true, false, true];
        let logits = vec![
            0.0, 1.0, 2.0, 5.0, 0.0, // -> 3
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, // tie -> first wins = 0
        ];
        let seq = denoise_step_sequential(&x_t, &mask, &logits, 5).unwrap();
        let par = denoise_step(&x_t, &mask, &logits, RemaskStrategy::None, 0.0, 5).unwrap();
        assert_eq!(seq.next_x, par.next_x);
        assert_eq!(seq.next_mask, par.next_mask);
    }

    #[test]
    fn accepts_valid_confidence_threshold() {
        let vocab = 2;
        let x_t = vec![0u32, 0];
        let mask = vec![true, true];
        let logits = vec![0.0, 10.0, 0.0, 10.0];
        let upd = denoise_step(&x_t, &mask, &logits, RemaskStrategy::None, 0.9, vocab).unwrap();
        // softmax([0,10]) -> max ≈ 1.0; both accepted.
        assert_eq!(upd.next_x, vec![1, 1]);
        assert_eq!(upd.accepted_count, 2);
    }

    #[test]
    fn rejects_invalid_confidence_threshold() {
        let vocab = 2;
        let x_t = vec![0u32, 0];
        let mask = vec![true, true];
        let logits = vec![0.0, 10.0, 0.0, 10.0];
        let err = denoise_step(&x_t, &mask, &logits, RemaskStrategy::None, 1.5, vocab).unwrap_err();
        assert!(matches!(err, KernelError::OutOfRange { .. }));
    }
}
