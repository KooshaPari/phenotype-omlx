//! Qwen3-Coder-Next sparse MoE pipeline acceptance.

use super::*;

// ===========================================================================
// Qwen3-Coder-Next sparse MoE acceptance
// ===========================================================================

#[test]
fn qwen_sparse_moe_pipeline_runs_end_to_end() {
    // 3 experts, top-2 router, 4 tokens, hidden=6, k=4.
    let num_experts = 3;
    let top_k = 2;
    let num_tokens = 4;
    let hidden = 6;
    let k = 4;
    let capacity_factor = 1.0; // capacity per expert = ceil(1.0 * 4 / 3) = 2

    // Per-token router logits (deterministic).
    let router_logits: Vec<f32> = (0..num_tokens)
        .map(|t| deterministic_vec(num_experts, SEED ^ (0xE0_01 + t as u64)))
        .flat_map(|v| v.into_iter())
        .collect();

    // Run the router and build the assignments per token.
    let mut assignments: Vec<(usize, f32)> = Vec::new();
    let mut picks_per_token: Vec<Vec<(usize, f32)>> = Vec::new();
    for t in 0..num_tokens {
        let logits = &router_logits[t * num_experts..(t + 1) * num_experts];
        let picks = router_topk(logits, num_experts, top_k, 0).unwrap();
        // The pipeline places tokens at the *first* top-k pick; we
        // keep the full picks list for the weighted reduce step.
        assignments.push(picks[0]);
        picks_per_token.push(picks);
    }

    // Dispatch: capacity_factor * num_tokens / num_experts -> 2 per
    // expert under cap=1.0.
    let token_indices: Vec<usize> = (0..num_tokens).collect();
    let plan: DispatchPlan = moe_dispatch(
        &token_indices,
        &assignments,
        num_experts,
        capacity_factor,
    )
    .unwrap();

    // Capacity-factor contract: no expert exceeds ceil(cap * n / E).
    let per_expert_cap = (capacity_factor * num_tokens as f32 / num_experts as f32).ceil() as usize;
    for (e, used) in plan.capacity_used.iter().enumerate() {
        assert!(*used <= per_expert_cap, "expert {e} used {used} > cap {per_expert_cap}");
    }
    let total_assigned: usize = plan.capacity_used.iter().sum();
    assert_eq!(total_assigned + plan.dropped.len(), num_tokens);

    // Shared expert: out[t, :] = x[t, :] @ w.
    let x = deterministic_vec(num_tokens * k, 0xA_CE);
    let w = deterministic_vec(k * hidden, 0xB_EE);
    let mut shared_out = vec![0.0f32; num_tokens * hidden];
    shared_expert(&x, &w, &mut shared_out).unwrap();

    // Reference: per-token dense matmul.
    let mut shared_ref = vec![0.0f32; num_tokens * hidden];
    for t in 0..num_tokens {
        for j in 0..hidden {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[t * k + kk] * w[kk * hidden + j];
            }
            shared_ref[t * hidden + j] = acc;
        }
    }
    assert_buf_close(&shared_out, &shared_ref, 1e-5, 1e-4);
    assert!(shared_out.iter().all(|v| v.is_finite()), "shared_out not finite");

    // weighted_reduce: shrink picks_per_token to its weights and
    // build a (num_tokens, top_k, hidden) expert_outs buffer by
    // expanding each pick through the same shared-expert matmul
    // (stand-in for the routed expert path; the goal of this trace
    // is to exercise the weighted_reduce contract).
    let mut expert_outs = vec![0.0f32; num_tokens * top_k * hidden];
    let mut weights = vec![0.0f32; num_tokens * top_k];
    for t in 0..num_tokens {
        for (e_idx, (expert, w_pick)) in picks_per_token[t].iter().enumerate() {
            // Per-(token, expert) pseudo-output: a simple linear
            // combination of x[t] and the expert index, masked into
            // the same hidden shape. This keeps the test free of a
            // full expert tensor while still exercising the
            // weighted_reduce shape contract.
            for j in 0..hidden {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += x[t * k + kk] * (((*expert as f32) + 1.0) * 0.1 + 0.05 * (j as f32) + 0.01 * (kk as f32));
                }
                expert_outs[(t * top_k + e_idx) * hidden + j] = acc;
            }
            weights[t * top_k + e_idx] = *w_pick;
        }
    }
    let mut reduced = vec![0.0f32; num_tokens * hidden];
    weighted_reduce(&expert_outs, &weights, top_k, hidden, &mut reduced).unwrap();
    assert!(reduced.iter().all(|v| v.is_finite()), "reduced not finite");

    // Sanity: the reduced output must equal the per-token weighted
    // sum of the per-(token, expert) outputs we just built.
    let mut expected_reduced = vec![0.0f32; num_tokens * hidden];
    for t in 0..num_tokens {
        for j in 0..hidden {
            let mut acc = 0.0f32;
            for e_idx in 0..top_k {
                let w_e = weights[t * top_k + e_idx];
                let v = expert_outs[(t * top_k + e_idx) * hidden + j];
                acc += w_e * v;
            }
            expected_reduced[t * hidden + j] = acc;
        }
    }
    assert_buf_close(&reduced, &expected_reduced, 1e-5, 1e-4);

    // End-to-end: every intermediate buffer touched in this trace
    // must be finite. The router picks are also tested for finite
    // weights (a single top-k weight must be > 0).
    for (t, picks) in picks_per_token.iter().enumerate() {
        for (e, w) in picks {
            assert!(w.is_finite() && *w > 0.0, "token {t} expert {e} weight not finite/positive");
        }
    }
}
