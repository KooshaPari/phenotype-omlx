//! MoE dispatch: assign tokens to expert buckets, enforce capacity,
//! route overflow to `dropped`.

use crate::error::{KernelError, Result};

/// Result of dispatching tokens to experts under a capacity constraint.
#[derive(Debug, Clone, PartialEq)]
pub struct DispatchPlan {
    /// `expert_buckets[e]` is the list of *original token indices* that
    /// expert `e` is responsible for (in input order).
    pub expert_buckets: Vec<Vec<usize>>,
    /// `capacity_used[e]` is the number of tokens placed in expert `e`.
    /// Always `<= ceil(capacity_factor * num_tokens / num_experts)`.
    pub capacity_used: Vec<usize>,
    /// Indices of tokens that exceeded every expert's capacity.
    pub dropped: Vec<usize>,
}

/// Build a [`DispatchPlan`] from per-token expert assignments.
///
/// `token_indices[i]` is the original token id for `assignments[i]`.
/// `assignments[i]` is `(expert_id, score)`.
///
/// The per-expert capacity is
/// `ceil(capacity_factor * num_tokens / num_experts)`. Tokens are placed
/// greedily in `assignments` order. If a token's expert is already at
/// capacity, the token is appended to `dropped`.
#[allow(clippy::too_many_arguments)]
pub fn moe_dispatch(
    token_indices: &[usize],
    assignments: &[(usize, f32)],
    num_experts: usize,
    capacity_factor: f32,
) -> Result<DispatchPlan> {
    if num_experts == 0 {
        return Err(KernelError::ZeroDimension {
            what: "num_experts",
            got: 0,
        });
    }
    if token_indices.len() != assignments.len() {
        return Err(KernelError::DimMismatch {
            what: "token_indices vs assignments",
            expected: assignments.len(),
            got: token_indices.len(),
        });
    }
    if !(capacity_factor > 0.0 && capacity_factor.is_finite()) {
        return Err(KernelError::BadCapacityFactor {
            got: capacity_factor,
        });
    }
    for &(e, _) in assignments {
        if e >= num_experts {
            return Err(KernelError::ExpertOutOfRange {
                num_experts,
                got: e,
            });
        }
    }
    let n = token_indices.len();
    let cap_exact = (capacity_factor * n as f32) / num_experts as f32;
    let capacity_per_expert = cap_exact.ceil() as usize;
    let mut plan = DispatchPlan {
        expert_buckets: vec![Vec::new(); num_experts],
        capacity_used: vec![0; num_experts],
        dropped: Vec::new(),
    };
    for (i, &(expert, _score)) in assignments.iter().enumerate() {
        let tok = token_indices[i];
        if plan.capacity_used[expert] < capacity_per_expert {
            plan.expert_buckets[expert].push(tok);
            plan.capacity_used[expert] += 1;
        } else {
            plan.dropped.push(tok);
        }
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_capacity_factor() {
        let err = moe_dispatch(&[], &[], 2, 0.0).unwrap_err();
        assert!(matches!(err, KernelError::BadCapacityFactor { .. }));
        let err = moe_dispatch(&[], &[], 2, -1.0).unwrap_err();
        assert!(matches!(err, KernelError::BadCapacityFactor { .. }));
    }

    #[test]
    fn rejects_expert_out_of_range() {
        let assignments = vec![(5usize, 0.5f32)];
        let err = moe_dispatch(&[0], &assignments, 4, 1.0).unwrap_err();
        assert!(matches!(err, KernelError::ExpertOutOfRange { .. }));
    }

    #[test]
    fn rejects_zero_num_experts() {
        let err = moe_dispatch(&[], &[], 0, 1.0).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let assignments = vec![(0usize, 0.5f32)];
        let err = moe_dispatch(&[], &assignments, 2, 1.0).unwrap_err();
        assert!(matches!(err, KernelError::DimMismatch { .. }));
    }

    #[test]
    fn capacity_factor_above_one_overallocates() {
        // 4 tokens, 2 experts, capacity_factor = 2.0 -> capacity 4 per
        // expert. All tokens land in expert 0.
        let tokens = vec![0, 1, 2, 3];
        let assignments = vec![(0, 0.5); 4];
        let plan = moe_dispatch(&tokens, &assignments, 2, 2.0).unwrap();
        assert_eq!(plan.expert_buckets[0].len(), 4);
        assert_eq!(plan.capacity_used[0], 4);
        assert!(plan.dropped.is_empty());
    }

    #[test]
    fn skewed_routing_reports_capacity_drops_deterministically() {
        let tokens = (0..8).collect::<Vec<_>>();
        let assignments = vec![(0, 1.0); tokens.len()];
        let plan = moe_dispatch(&tokens, &assignments, 4, 1.0).unwrap();

        assert_eq!(plan.capacity_used, vec![2, 0, 0, 0]);
        assert_eq!(plan.expert_buckets[0], vec![0, 1]);
        assert_eq!(plan.dropped, vec![2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn balanced_routing_avoids_drops_at_unit_capacity() {
        let tokens = (0..8).collect::<Vec<_>>();
        let assignments = (0..8).map(|token| (token % 4, 1.0)).collect::<Vec<_>>();
        let plan = moe_dispatch(&tokens, &assignments, 4, 1.0).unwrap();

        assert_eq!(plan.capacity_used, vec![2, 2, 2, 2]);
        assert!(plan.dropped.is_empty());
    }
}
