//! Pure remask logic for the diffusion scheduler.
//!
//! [`remask`] takes a per-position confidence score vector and writes a
//! re-mask decision into a pre-existing boolean mask under the chosen
//! [`RemaskStrategy`]. All variants are deterministic given the input
//! scores and the `(step, total_steps)` pair.

use crate::common::Lcg;
use crate::error::{KernelError, Result};

use super::denoise::RemaskStrategy;

/// Pure remask logic. Given current confidence scores, decide which
/// positions should be re-masked. `mask` is overwritten in place:
/// after the call, `mask[i] = true` iff position `i` should be
/// re-masked under the chosen strategy.
///
/// `step` and `total_steps` are forwarded into the seeded RNG for
/// `RandomFraction` so the call is fully deterministic.
pub fn remask(
    scores: &[f32],
    mask: &mut [bool],
    strategy: &RemaskStrategy,
    step: usize,
    total_steps: usize,
) -> Result<()> {
    let n = scores.len();
    if mask.len() != n {
        return Err(KernelError::DimMismatch {
            what: "remask.scores vs mask",
            expected: n,
            got: mask.len(),
        });
    }
    if total_steps == 0 {
        return Err(KernelError::BadBufferLength {
            what: "total_steps",
            expected: 1,
            got: 0,
        });
    }
    if let Some(index) = scores.iter().position(|score| !score.is_finite()) {
        return Err(KernelError::NonFiniteValue {
            what: "remask.scores",
            index,
        });
    }
    match strategy {
        RemaskStrategy::None => {
            // Leave the input mask untouched.
            Ok(())
        }
        RemaskStrategy::LowConfidence { percentile } => {
            if !(*percentile >= 0.0 && *percentile <= 100.0) {
                return Err(KernelError::OutOfRange {
                    what: "percentile",
                    min: 0.0,
                    max: 100.0,
                    got: *percentile,
                });
            }
            if scores.is_empty() {
                return Ok(());
            }
            if *percentile == 100.0 {
                mask.fill(true);
                return Ok(());
            }
            // Sort scores to find the threshold at `percentile`.
            let mut sorted = scores.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let norm = (percentile / 100.0).clamp(0.0, 1.0);
            let idx = (norm * (sorted.len() as f32 - 1.0)).round() as usize;
            let idx = idx.min(sorted.len() - 1);
            let threshold = sorted[idx];
            // Re-mask positions whose confidence is strictly below
            // the percentile threshold. Positions at or above the
            // threshold are accepted.
            for (i, &s) in scores.iter().enumerate() {
                mask[i] = s < threshold;
            }
            Ok(())
        }
        RemaskStrategy::EntropyBased => {
            if scores.is_empty() {
                return Ok(());
            }
            // Treat low-decisiveness (low confidence) as high
            // entropy and re-mask below the 25th percentile of
            // confidence.
            let mut sorted = scores.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = (sorted.len() as f32 * 0.25).floor() as usize;
            let idx = idx.min(sorted.len().saturating_sub(1));
            let threshold = sorted[idx];
            for (i, &s) in scores.iter().enumerate() {
                mask[i] = s < threshold;
            }
            Ok(())
        }
        RemaskStrategy::RandomFraction(f) => {
            if !(*f >= 0.0 && *f <= 1.0) {
                return Err(KernelError::OutOfRange {
                    what: "RandomFraction",
                    min: 0.0,
                    max: 1.0,
                    got: *f,
                });
            }
            // Deterministic seeded RNG: same `(step, total_steps)`
            // yields the same mask outcome.
            let seed = (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (total_steps as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let mut rng = Lcg::new(seed);
            // Start from all-false (nothing re-masked), then flip on
            // the chosen positions.
            for m in mask.iter_mut() {
                *m = false;
            }
            let mut indices: Vec<usize> = (0..n).collect();
            let k = ((f * n as f32).round() as usize).min(n);
            for i in 0..k {
                let span = indices.len() - i;
                if span == 0 {
                    break;
                }
                let j = (rng.next_u64() as usize) % span + i;
                indices.swap(i, j);
            }
            for &i in &indices[..k] {
                mask[i] = true;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_leaves_mask_unchanged() {
        let mut mask = vec![false, false, false];
        remask(&[0.9, 0.8, 0.7], &mut mask, &RemaskStrategy::None, 0, 1).unwrap();
        assert_eq!(mask, vec![false, false, false]);
    }

    #[test]
    fn low_confidence_percentile_fifty_masks_lower_half() {
        let mut mask = vec![true, true, true, true];
        remask(
            &[0.9, 0.8, 0.2, 0.1],
            &mut mask,
            &RemaskStrategy::LowConfidence { percentile: 50.0 },
            0,
            1,
        )
        .unwrap();
        // Sorted: [0.1, 0.2, 0.8, 0.9]; median index = round(0.5 * 3) = 2 -> 0.8.
        // Anything strictly less than 0.8 is re-masked -> tokens 2, 3.
        assert_eq!(mask, vec![false, false, true, true]);
    }

    #[test]
    fn low_confidence_percentile_zero_keeps_all() {
        let mut mask = vec![false, false, false];
        remask(
            &[0.9, 0.5, 0.3],
            &mut mask,
            &RemaskStrategy::LowConfidence { percentile: 0.0 },
            0,
            1,
        )
        .unwrap();
        assert_eq!(mask, vec![false, false, false]);
    }

    #[test]
    fn low_confidence_percentile_hundred_masks_all() {
        let mut mask = vec![false, false, false];
        remask(
            &[0.9, 0.5, 0.3],
            &mut mask,
            &RemaskStrategy::LowConfidence { percentile: 100.0 },
            0,
            1,
        )
        .unwrap();
        assert_eq!(mask, vec![true, true, true]);
    }

    #[test]
    fn rejects_nonfinite_scores_before_remasking() {
        let mut mask = vec![false, false, false];
        let err = remask(
            &[f32::NAN, 0.5, f32::INFINITY],
            &mut mask,
            &RemaskStrategy::LowConfidence { percentile: 100.0 },
            0,
            1,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            KernelError::NonFiniteValue {
                what: "remask.scores",
                ..
            }
        ));
    }

    #[test]
    fn random_fraction_is_deterministic() {
        let scores = vec![0.5f32; 100];
        let mut m1 = vec![false; 100];
        let mut m2 = vec![false; 100];
        remask(
            &scores,
            &mut m1,
            &RemaskStrategy::RandomFraction(0.5),
            3,
            10,
        )
        .unwrap();
        remask(
            &scores,
            &mut m2,
            &RemaskStrategy::RandomFraction(0.5),
            3,
            10,
        )
        .unwrap();
        assert_eq!(m1, m2);
        let count = m1.iter().filter(|b| **b).count();
        assert_eq!(count, 50);
    }

    #[test]
    fn random_fraction_with_zero_masks_nothing() {
        let scores = vec![0.5f32; 10];
        let mut mask = vec![false; 10];
        remask(
            &scores,
            &mut mask,
            &RemaskStrategy::RandomFraction(0.0),
            0,
            1,
        )
        .unwrap();
        assert!(mask.iter().all(|b| !*b));
    }

    #[test]
    fn rejects_invalid_percentile() {
        let mut mask = vec![false, false];
        let err = remask(
            &[0.5, 0.5],
            &mut mask,
            &RemaskStrategy::LowConfidence { percentile: 150.0 },
            0,
            1,
        )
        .unwrap_err();
        assert!(matches!(err, KernelError::OutOfRange { .. }));
    }

    #[test]
    fn entropy_based_is_decisive() {
        let mut mask = vec![false, false, false, false];
        remask(
            &[0.9, 0.5, 0.05, 0.1],
            &mut mask,
            &RemaskStrategy::EntropyBased,
            0,
            1,
        )
        .unwrap();
        // Sorted: [0.05, 0.1, 0.5, 0.9]; 25th-percentile index = 0 -> 0.05.
        // Anything <= 0.05 is re-masked -> position 2 only.
        assert_eq!(mask, vec![false, false, true, false]);
    }
}
