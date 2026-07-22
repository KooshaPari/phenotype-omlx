//! Perplexity scoring for the eval harness.
//!
//! Perplexity is a calibration-free quality metric computed from per-token
//! negative log-likelihoods. The function is pure, deterministic, and
//! allocation-light. No dataset is bundled; callers supply a sequence of
//! log-probabilities from their own model.

use crate::{Suite, TaskSpec};

/// Compute perplexity from per-token log-probabilities.
///
/// Returns positive infinity for an empty input so callers can distinguish
/// "no tokens" from a degenerate perplexity of 1.0 (every token predicted with
/// probability 1). Non-finite inputs propagate to NaN/infinity per IEEE-754.
pub fn score_perplexity(log_probs: &[f64]) -> f64 {
    if log_probs.is_empty() {
        return f64::INFINITY;
    }
    let nll: f64 = log_probs.iter().sum();
    (-nll / log_probs.len() as f64).exp()
}

/// Construct a caller-identified task descriptor for a perplexity run.
pub fn task_descriptor(id: impl Into<String>, prompt: impl Into<String>) -> TaskSpec {
    TaskSpec {
        id: id.into(),
        suite: Suite::Perplexity,
        prompt: prompt.into(),
        expected: None,
        choices: vec![],
        criteria: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_log_probs_yields_infinity() {
        assert!(score_perplexity(&[]).is_infinite());
        assert!(score_perplexity(&[]) > 0.0);
    }

    #[test]
    fn uniform_log_probability_is_e_to_minus_log_p() {
        // All tokens have log-prob = log(0.5), so PPL = exp(-log(0.5)) = 2.
        let log_probs = vec![0.5_f64.ln(); 4];
        assert!((score_perplexity(&log_probs) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn perfect_predictions_yield_ppl_one() {
        // log(1.0) = 0, so -mean(log p) = 0, exp(0) = 1.
        let log_probs = vec![0.0_f64; 8];
        assert!((score_perplexity(&log_probs) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn deterministic_for_same_input() {
        let log_probs = vec![-0.1, -0.5, -1.2, -2.0];
        let a = score_perplexity(&log_probs);
        let b = score_perplexity(&log_probs);
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn task_descriptor_carries_identity_prompt_and_suite() {
        let t = task_descriptor("wiki-test-17", "The quick brown fox.");
        assert_eq!(t.id, "wiki-test-17");
        assert_eq!(t.suite, Suite::Perplexity);
        assert_eq!(t.prompt, "The quick brown fox.");
        assert_eq!(t.expected, None);
        assert!(t.choices.is_empty());
        assert!(t.criteria.is_none());
    }
}
