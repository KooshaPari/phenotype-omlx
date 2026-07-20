use super::*;

fn cfg(capacity_factor: f32) -> ModRouterConfig {
    ModRouterConfig {
        capacity_factor,
        mean_capacity: 0.0,
    }
}

#[test]
fn capacity_one_returns_all_tokens_in_score_descending_order() {
    // Sigmoid is monotonic, so the score-desc order over all
    // tokens is the same as weight-desc order. With weights
    // [0.1, -0.2, 0.3, -0.4, 0.5] the top-scoring tokens are 4
    // (w=0.5), 2 (w=0.3), 0 (w=0.1), 1 (w=-0.2), 3 (w=-0.4).
    let weights = vec![0.1f32, -0.2, 0.3, -0.4, 0.5];
    let plan = mod_route(&weights, &cfg(1.0)).unwrap();
    assert_eq!(plan.selected_tokens, vec![4, 2, 0, 1, 3]);
}

#[test]
fn capacity_half_returns_top_half_by_sigmoid_score() {
    // Weights are monotonically increasing -> sigmoid scores are
    // strictly increasing -> top half is the high-weight tail.
    let weights: Vec<f32> = (0..10).map(|i| (i as f32 - 4.5) * 2.0).collect();
    let plan = mod_route(&weights, &cfg(0.5)).unwrap();
    assert_eq!(plan.selected_tokens.len(), 5);
    // Strictly increasing scores => top 5 are the last 5 indices,
    // listed score-descending (which equals index-descending here).
    assert_eq!(plan.selected_tokens, vec![9, 8, 7, 6, 5]);
}

#[test]
fn deterministic_across_runs() {
    let weights = vec![0.2f32, 0.9, -0.4, 0.7, 0.0, 0.5];
    let p1 = mod_route(&weights, &cfg(0.5)).unwrap();
    let p2 = mod_route(&weights, &cfg(0.5)).unwrap();
    assert_eq!(p1, p2);
}

#[test]
fn empty_weights_yields_empty_plan() {
    let plan = mod_route(&[], &cfg(0.5)).unwrap();
    assert!(plan.selected_tokens.is_empty());
    assert_eq!(plan.capacity_factor, 0.5);
}

#[test]
fn ties_break_by_lower_index() {
    // Identical weights -> identical sigmoid scores -> sort by
    // index ascending among ties.
    let weights = vec![0.5f32; 6];
    let plan = mod_route(&weights, &cfg(0.5)).unwrap();
    assert_eq!(plan.selected_tokens, vec![0, 1, 2]);
}

#[test]
fn apply_scatter_round_trip_with_zero_fill_is_exact() {
    let weights = vec![0.1f32, 0.5, -0.2, 0.9, 0.0, 0.3];
    let plan = mod_route(&weights, &cfg(0.5)).unwrap();
    let num_tokens = weights.len();
    let dim = 3;
    // Build a deterministic full buffer: hidden[t, d] = t * 10 + d.
    let full: Vec<f32> = (0..num_tokens * dim)
        .map(|i| (i / dim) as f32 * 10.0 + (i % dim) as f32)
        .collect();
    // Apply: extract the surviving rows.
    let selected = mod_apply(&plan, &full, dim).unwrap();
    // Scatter back with fill=0.0; the result must match `full` at
    // surviving rows and be 0.0 at the rest.
    let mut scattered = mod_scatter_back(&selected, &plan, num_tokens, dim, 0.0).unwrap();
    // Zero out the non-survivor rows on a reference copy.
    let mut expected = full.clone();
    let survivor: std::collections::HashSet<u32> =
        plan.selected_tokens.iter().copied().collect();
    for t in 0..num_tokens {
        if !survivor.contains(&(t as u32)) {
            for d in 0..dim {
                expected[t * dim + d] = 0.0;
            }
        }
    }
    // L2 sum of squared diffs must be 0.
    let l2: f64 = scattered
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| {
            let d = (*a as f64) - (*b as f64);
            d * d
        })
        .sum();
    assert_eq!(l2, 0.0, "round-trip should be exact with fill=0.0; got L2={l2}");
    // Sanity: the surviving rows really are the original rows.
    for &idx in &plan.selected_tokens {
        let row = idx as usize;
        for d in 0..dim {
            assert_eq!(scattered[row * dim + d], full[row * dim + d]);
        }
    }
    // Defensive: scattered must be the same length as the full
    // buffer.
    assert_eq!(scattered.len(), full.len());
    // Silence unused-mut on scattered (the binding is reassigned
    // by .unwrap() above but the variable would be flagged if we
    // didn't touch it again).
    let _ = &mut scattered;
}

#[test]
fn rejects_capacity_factor_outside_unit_interval() {
    let weights = vec![0.1f32, 0.2, 0.3];
    // Zero: must reject.
    let err = mod_route(&weights, &cfg(0.0)).unwrap_err();
    assert!(matches!(err, KernelError::OutOfRange { .. }));
    // Negative: must reject.
    let err = mod_route(&weights, &cfg(-0.1)).unwrap_err();
    assert!(matches!(err, KernelError::OutOfRange { .. }));
    // Above 1: must reject.
    let err = mod_route(&weights, &cfg(1.5)).unwrap_err();
    assert!(matches!(err, KernelError::OutOfRange { .. }));
    // NaN: must reject.
    let err = mod_route(&weights, &cfg(f32::NAN)).unwrap_err();
    assert!(matches!(err, KernelError::OutOfRange { .. }));
}

#[test]
fn capacity_factor_one_is_accepted_and_returns_all_tokens() {
    // Strictly increasing weights => score-desc order is the same
    // as index-desc order, so all tokens appear, in reverse index
    // order.
    let weights = vec![0.1f32, 0.2, 0.3, 0.4];
    let plan = mod_route(&weights, &cfg(1.0)).unwrap();
    assert_eq!(plan.selected_tokens, vec![3, 2, 1, 0]);
    assert_eq!(plan.capacity_factor, 1.0);
}

#[test]
fn very_small_capacity_promotes_at_least_one_survivor() {
    // capacity_factor * num_tokens floors to 0; the kernel must
    // promote k to 1 so the layer still has work to do.
    let weights = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
    let plan = mod_route(&weights, &cfg(0.0001)).unwrap();
    assert_eq!(plan.selected_tokens.len(), 1);
}

#[test]
fn scatter_rejects_mismatched_selected_length() {
    let plan = ModRoutePlan {
        selected_tokens: vec![0, 2],
        capacity_factor: 0.5,
    };
    let dim = 3;
    // Selected has the wrong length (would need 6 elements for 2 rows of 3).
    let bad = vec![0.0f32; 5];
    let err = mod_scatter_back(&bad, &plan, 4, dim, 0.0).unwrap_err();
    assert!(matches!(err, KernelError::BadBufferLength { .. }));
}

#[test]
fn apply_rejects_zero_dim() {
    let plan = ModRoutePlan {
        selected_tokens: vec![0],
        capacity_factor: 1.0,
    };
    let err = mod_apply(&plan, &[0.0, 0.0, 0.0], 0).unwrap_err();
    assert!(matches!(err, KernelError::ZeroDimension { .. }));
}
