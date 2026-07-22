//! Qwen agentic end-to-end mini-trace: DeltaNet + sparse MoE composition.

use super::*;

// ===========================================================================
// Qwen3-Coder-Next agentic mini-trace (DeltaNet + sparse MoE composition)
// ===========================================================================

#[test]
fn qwen_agentic_mini_trace_runs_and_is_finite() {
    // Glue the DeltaNet trace and the MoE pipeline together.
    // The DeltaNet outputs per step, per head are stacked into a
    // single hidden vector; the first `hidden` columns of each
    // token-row drive the MoE shared-expert path. The intent is to
    // assert that all buffers touched end-to-end are finite and
    // that the public API contracts (router / dispatch / shared /
    // reduce) compose without panic.
    let num_heads = 4;
    let head_dim = 4;
    let chunk_size = 16;
    let hidden = num_heads * head_dim; // 16
    let num_tokens = 4;
    let num_experts = 3;
    let top_k = 2;

    // DeltaNet inputs and per-head states.
    let q = deterministic_vec(chunk_size * num_heads * head_dim, 0xCC_AA);
    let kd = deterministic_vec(chunk_size * num_heads * head_dim, 0xCC_BB);
    let v = deterministic_vec(chunk_size * num_heads * head_dim, 0xCC_CC);
    let initial_states: Vec<Vec<f32>> = (0..num_heads)
        .map(|h| {
            let mut rng = Lcg::new(SEED ^ (0xCC_E0u64 + h as u64));
            (0..head_dim * head_dim).map(|_| rng.next_signed() * 0.25).collect()
        })
        .collect();
    let (deltanet_outs, deltanet_states) = run_qwen_deltanet_trace(
        &q,
        &kd,
        &v,
        &initial_states,
        chunk_size,
        num_heads,
        head_dim,
    );
    assert!(deltanet_outs.iter().all(|v| v.is_finite()));
    for (h, state) in deltanet_states.iter().enumerate().take(num_heads) {
        assert!(
            state.iter().all(|v| v.is_finite()),
            "head {h} state not finite"
        );
    }

    // Reduce the per-head, per-step outputs into per-token
    // activations: take the *first* `hidden` columns of every
    // chunk_size row and split it into `num_tokens` tokens of
    // length `hidden`. (This is a structural stand-in for a real
    // projection; it exists only to exercise the MoE downstream.)
    let token_x: Vec<f32> = (0..num_tokens)
        .flat_map(|t| {
            let start = (t * (chunk_size / num_tokens)) * hidden;
            deltanet_outs[start..start + hidden].iter().copied()
        })
        .collect();
    assert_eq!(token_x.len(), num_tokens * hidden);
    assert!(token_x.iter().all(|v| v.is_finite()));

    // MoE: per-token router over `num_experts` slots.
    let mut all_router_logits = Vec::new();
    for t in 0..num_tokens {
        all_router_logits.extend(deterministic_vec(num_experts, SEED ^ (0xCC_D0 + t as u64)));
    }
    let mut assignments = Vec::new();
    for t in 0..num_tokens {
        let logits = &all_router_logits[t * num_experts..(t + 1) * num_experts];
        let picks = router_topk(logits, num_experts, top_k, 0).unwrap();
        assignments.push(picks[0]);
    }
    let plan = moe_dispatch(&(0..num_tokens).collect::<Vec<_>>(), &assignments, num_experts, 1.0).unwrap();

    // Shared expert: project token_x through a [hidden, hidden]
    // weight matrix.
    let w = deterministic_vec(hidden * hidden, 0xCC_EE);
    let mut shared_out = vec![0.0f32; num_tokens * hidden];
    shared_expert(&token_x, &w, &mut shared_out).unwrap();
    assert!(shared_out.iter().all(|v| v.is_finite()));

    // weighted_reduce with a (num_tokens, top_k, hidden) tensor of
    // shared-expert-like pseudo outputs to keep the pipeline alive.
    let mut expert_outs = vec![0.0f32; num_tokens * top_k * hidden];
    let mut weights = vec![0.0f32; num_tokens * top_k];
    for t in 0..num_tokens {
        for e_idx in 0..top_k {
            for j in 0..hidden {
                expert_outs[(t * top_k + e_idx) * hidden + j] = shared_out[t * hidden + j]
                    * (1.0 + 0.1 * (e_idx as f32));
            }
            weights[t * top_k + e_idx] = 1.0 / top_k as f32;
        }
    }
    let mut reduced = vec![0.0f32; num_tokens * hidden];
    weighted_reduce(&expert_outs, &weights, top_k, hidden, &mut reduced).unwrap();
    assert!(reduced.iter().all(|v| v.is_finite()), "reduced not finite");

    // The trace touched the capacity plan (verify it ran without
    // error); the bucket lengths must equal the number of experts.
    assert_eq!(plan.expert_buckets.len(), num_experts);
    // And `approx_eq` is reachable through this trace as a smoke
    // test (avoids unused-import warnings on `approx_eq`).
    assert!(approx_eq(reduced[0], reduced[0]));
}
