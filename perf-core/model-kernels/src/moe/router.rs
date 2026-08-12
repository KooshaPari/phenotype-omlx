//! MoE top-k router with deterministic tie-breaking.

use crate::common::Lcg;
use crate::error::{KernelError, Result};

/// Pick `top_k` experts for a single token. Returns the picks in
/// `(expert_id, renormalized_weight)` order, sorted by score descending
/// (ties broken deterministically by expert id ascending).
///
/// `tie_break_seed` is forwarded into the deterministic LCG. Callers
/// that do not need seed-driven shuffling can pass any value (e.g. `0`).
pub fn router_topk(
    router_logits: &[f32],
    num_experts: usize,
    top_k: usize,
    tie_break_seed: u64,
) -> Result<Vec<(usize, f32)>> {
    if num_experts == 0 {
        return Err(KernelError::ZeroDimension {
            what: "num_experts",
            got: 0,
        });
    }
    if router_logits.len() != num_experts {
        return Err(KernelError::BadBufferLength {
            what: "router_logits",
            expected: num_experts,
            got: router_logits.len(),
        });
    }
    if let Some(index) = router_logits.iter().position(|logit| !logit.is_finite()) {
        return Err(KernelError::NonFiniteValue {
            what: "router_logits",
            index,
        });
    }
    if top_k == 0 {
        return Err(KernelError::ZeroDimension {
            what: "top_k",
            got: 0,
        });
    }
    if top_k > num_experts {
        return Err(KernelError::DimMismatch {
            what: "top_k vs num_experts",
            expected: num_experts,
            got: top_k,
        });
    }
    let mut indexed: Vec<(usize, f32)> = router_logits.iter().copied().enumerate().collect();
    // Sort by (score DESC, expert_id ASC). Using reversed sort key.
    indexed.sort_by(|(ea, la), (eb, lb)| {
        lb.partial_cmp(la)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ea.cmp(eb))
    });
    let picks: Vec<(usize, f32)> = indexed.iter().take(top_k).map(|(e, l)| (*e, *l)).collect();
    // Renormalize via softmax over the top-k logits (numerically stable).
    let max = picks
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = picks.iter().map(|(_, l)| (*l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let mut rng = Lcg::new(tie_break_seed);
    // For exact ties on (score, expert_id) the above sort is already
    // deterministic; we still consume one rng draw per pick so that the
    // function's randomness surface is documented and stable.
    let mut out = Vec::with_capacity(top_k);
    for (i, (e, _)) in picks.iter().enumerate() {
        let w = if sum > 0.0 {
            exp[i] / sum
        } else {
            1.0 / top_k as f32
        };
        let _ = rng.next_u64();
        out.push((*e, w));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_num_experts() {
        let err = router_topk(&[], 0, 1, 0).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn rejects_top_k_greater_than_num_experts() {
        let logits = [1.0f32, 2.0];
        let err = router_topk(&logits, 2, 3, 0).unwrap_err();
        assert!(matches!(err, KernelError::DimMismatch { .. }));
    }

    #[test]
    fn picks_top_by_score() {
        let logits = [1.0f32, 5.0, 2.0, 3.0];
        let picks = router_topk(&logits, 4, 2, 0).unwrap();
        assert_eq!(picks.len(), 2);
        // Highest logit is index 1, then 3.
        assert_eq!(picks[0].0, 1);
        assert_eq!(picks[1].0, 3);
    }

    #[test]
    fn weights_are_positive_and_sum_to_one() {
        let logits = [1.0f32, 5.0, 2.0, 3.0];
        let picks = router_topk(&logits, 4, 2, 0).unwrap();
        let s: f32 = picks.iter().map(|(_, w)| *w).sum();
        assert!((s - 1.0).abs() < 1e-5);
        for (_, w) in picks {
            assert!(w > 0.0);
        }
    }

    #[test]
    fn rejects_non_finite_logits_before_sorting() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = router_topk(&[0.0, bad, 1.0], 3, 2, 0).unwrap_err();
            assert!(matches!(
                err,
                KernelError::NonFiniteValue {
                    what: "router_logits",
                    index: 1
                }
            ));
        }
    }
}
