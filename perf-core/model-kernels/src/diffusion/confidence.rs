//! Per-token softmax-max confidence scores.

use crate::common::softmax_max;
use crate::error::{KernelError, Result};

/// Compute confidence scores for `vocab`-wide logits.
///
/// `logits` is laid out row-major as `[num_tokens, vocab]`. The
/// returned vector has length `num_tokens`; each entry is the
/// softmax-max of the corresponding row (i.e. the model's confidence
/// in its own argmax for that position).
///
/// `vocab` must be a power of two. This matches the standard
/// matmul-tile shapes used elsewhere in the kernel crate and lets
/// the caller pre-pad logits to a known size without surprises.
pub fn confidence_scores(logits: &[f32], vocab: usize) -> Result<Vec<f32>> {
    if vocab == 0 {
        return Err(KernelError::ZeroDimension {
            what: "vocab",
            got: 0,
        });
    }
    if !vocab.is_power_of_two() {
        return Err(KernelError::BadBufferLength {
            what: "vocab (must be power of two)",
            expected: vocab.next_power_of_two(),
            got: vocab,
        });
    }
    if logits.is_empty() {
        return Ok(Vec::new());
    }
    if logits.len() % vocab != 0 {
        return Err(KernelError::BadBufferLength {
            what: "logits",
            expected: vocab,
            got: logits.len(),
        });
    }
    let n = logits.len() / vocab;
    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let row = &logits[i * vocab..(i + 1) * vocab];
        scores.push(softmax_max(row));
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_vocab() {
        let err = confidence_scores(&[], 0).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn rejects_non_power_of_two_vocab() {
        let err = confidence_scores(&[1.0, 2.0, 3.0], 3).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn empty_logits_returns_empty() {
        let s = confidence_scores(&[], 4).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn softmax_max_matches_hand_computed() {
        // Single row: logits = [1, 2, 5, 0] with vocab = 4.
        let logits = [1.0f32, 2.0, 5.0, 0.0];
        let s = confidence_scores(&logits, 4).unwrap();
        assert_eq!(s.len(), 1);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: f32 = logits.iter().map(|l| (l - max).exp()).sum();
        let expected = (5.0f32 - max).exp() / exp;
        assert!((s[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn multi_row_logits_one_score_per_row() {
        let logits = vec![
            1.0, 2.0, 5.0, 0.0, // row 0
            0.0, 1.0, 2.0, 3.0, // row 1
        ];
        let s = confidence_scores(&logits, 4).unwrap();
        assert_eq!(s.len(), 2);
        assert!(s[0] > 0.5); // row 0 has clear argmax at index 2
        assert!(s[1] > 0.3); // row 1 has argmax at index 3
    }

    #[test]
    fn rejects_logits_length_not_multiple_of_vocab() {
        let err = confidence_scores(&[1.0, 2.0, 3.0, 4.0, 5.0], 4).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }
}
