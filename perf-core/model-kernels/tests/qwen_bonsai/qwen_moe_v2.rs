//! Qwen3-Coder-Next sparse-MoE per-stage composition (v2):
//! `router -> dispatch -> grouped_gemm_tiled -> weighted_reduce_tiled ->
//! stage_expert_outputs + coalesced_writeback`, with a shared-expert
//! projection running in parallel.
//!
//! Trace shape (deterministic via the crate-local `SEED` XOR salts):
//!
//! - `num_tokens = 4`, `num_experts = 3`, `top_k = 2`, `hidden = 4`,
//!   `k = 4` (intermediate dim), `capacity_factor = 2.0`
//!   (capacity = ceil(2.0 * 4 / 3) = 3 per expert).
//! - Router logits LCG-seeded per-token with salt
//!   `SEED ^ (0xE0_01 + t as u64)`.
//! - Shared expert weight W LCG-seeded with salt `SEED ^ 0xB_EE`,
//!   length `hidden * hidden`.
//! - Per-expert routed weights B[e] LCG-seeded with salt
//!   `SEED ^ (0xB0_E0 + e as u64)`, shape `[k, hidden]` per expert.
//! - Activations `a` LCG-seeded with salt `SEED ^ 0xA_CE`,
//!   length `num_tokens * k`.
//!
//! Asserts:
//! 1. Full pipeline runs end-to-end; residual buffer matches a
//!    hand-rolled scalar reference row-by-row (byte-equality).
//! 2. `grouped_gemm_tiled` is byte-equal to scalar `grouped_gemm`
//!    on the same dispatch plan.
//! 3. `weighted_reduce_tiled` is byte-equal to scalar `weighted_reduce`
//!    on the same `(expert_outs, weights)` inputs.

use super::*;
use model_kernels::moe::{
    coalesced_writeback, grouped_gemm, grouped_gemm_tiled, moe_dispatch, router_topk,
    shared_expert, stage_expert_outputs, weighted_reduce, weighted_reduce_tiled, DispatchPlan,
};

const NUM_TOKENS: usize = 4;
const NUM_EXPERTS: usize = 3;
const TOP_K: usize = 2;
const HIDDEN: usize = 4;
const K_INTERMEDIATE: usize = 4;
const CAPACITY_FACTOR: f32 = 2.0;

fn build_router_logits() -> Vec<f32> {
    (0..NUM_TOKENS)
        .flat_map(|t| deterministic_vec(NUM_EXPERTS, 0xE0_01 + t as u64))
        .collect()
}

#[allow(clippy::type_complexity)]
fn run_router(router_logits: &[f32]) -> (Vec<(usize, f32)>, Vec<Vec<(usize, f32)>>) {
    let mut assignments: Vec<(usize, f32)> = Vec::with_capacity(NUM_TOKENS);
    let mut picks_per_token: Vec<Vec<(usize, f32)>> = Vec::with_capacity(NUM_TOKENS);
    for t in 0..NUM_TOKENS {
        let logits = &router_logits[t * NUM_EXPERTS..(t + 1) * NUM_EXPERTS];
        let picks = router_topk(logits, NUM_EXPERTS, TOP_K, 0)
            .expect("router_topk must accept well-formed inputs");
        // Dispatch places tokens at the *first* top-k pick; the full
        // picks list feeds the weighted-reduce step.
        assignments.push(picks[0]);
        picks_per_token.push(picks);
    }
    (assignments, picks_per_token)
}

/// Build `[NUM_TOKENS, TOP_K, HIDDEN]` expert-outs. Slot 0 per token
/// comes from `grouped_gemm_tiled` over the dispatch plan (so the
/// trace exercises the tiled path); slot 1 per token is the scalar
/// matmul for the second top-k pick (which the dispatch plan does
/// not route).
fn build_expert_outs(
    a: &[f32],
    b: &[f32],
    plan: &DispatchPlan,
    picks_per_token: &[Vec<(usize, f32)>],
) -> Vec<f32> {
    let mut routed = vec![0.0f32; NUM_TOKENS * HIDDEN];
    grouped_gemm_tiled(
        a,
        b,
        &plan.expert_buckets,
        0,
        K_INTERMEDIATE,
        HIDDEN,
        &mut routed,
    )
    .expect("grouped_gemm_tiled must accept well-formed inputs");
    let mut expert_outs = vec![0.0f32; NUM_TOKENS * TOP_K * HIDDEN];
    for t in 0..NUM_TOKENS {
        for j in 0..HIDDEN {
            expert_outs[(t * TOP_K) * HIDDEN + j] = routed[t * HIDDEN + j];
        }
        let expert_e = picks_per_token[t][1].0;
        let b_offset = expert_e * K_INTERMEDIATE * HIDDEN;
        for j in 0..HIDDEN {
            let mut acc = 0.0f32;
            for kk in 0..K_INTERMEDIATE {
                acc += a[t * K_INTERMEDIATE + kk] * b[b_offset + kk * HIDDEN + j];
            }
            expert_outs[(t * TOP_K + 1) * HIDDEN + j] = acc;
        }
    }
    expert_outs
}

fn build_weights(picks_per_token: &[Vec<(usize, f32)>]) -> Vec<f32> {
    let mut weights = vec![0.0f32; NUM_TOKENS * TOP_K];
    for t in 0..NUM_TOKENS {
        for (e_idx, (_, w)) in picks_per_token[t].iter().enumerate() {
            weights[t * TOP_K + e_idx] = *w;
        }
    }
    weights
}

/// (a) End-to-end: full pipeline runs and the residual buffer
/// matches a hand-rolled scalar reference row-by-row.
#[test]
fn qwen_moe_v2_pipeline_runs_end_to_end_with_tiled_kernels() {
    let router_logits = build_router_logits();
    let (assignments, picks_per_token) = run_router(&router_logits);

    let token_indices: Vec<usize> = (0..NUM_TOKENS).collect();
    let plan: DispatchPlan = moe_dispatch(
        &token_indices,
        &assignments,
        NUM_EXPERTS,
        CAPACITY_FACTOR,
    )
    .expect("dispatch must accept well-formed inputs");

    // Sanity: no expert exceeds ceil(2.0 * 4 / 3) = 3.
    let per_expert_cap =
        (CAPACITY_FACTOR * NUM_TOKENS as f32 / NUM_EXPERTS as f32).ceil() as usize;
    for (e, used) in plan.capacity_used.iter().enumerate() {
        assert!(
            *used <= per_expert_cap,
            "expert {e} used {used} > cap {per_expert_cap}"
        );
    }
    let total_routed: usize = plan.capacity_used.iter().sum();
    assert_eq!(total_routed + plan.dropped.len(), NUM_TOKENS);

    // Activations and weights.
    let a = deterministic_vec(NUM_TOKENS * K_INTERMEDIATE, 0xA_CE);
    let w = deterministic_vec(HIDDEN * HIDDEN, 0xB_EE);
    let b: Vec<f32> = (0..NUM_EXPERTS)
        .flat_map(|e| deterministic_vec(K_INTERMEDIATE * HIDDEN, 0xB0_E0 + e as u64))
        .collect();

    // Shared-expert matmul.
    let mut shared_out = vec![0.0f32; NUM_TOKENS * HIDDEN];
    shared_expert(&a, &w, &mut shared_out)
        .expect("shared_expert must accept well-formed inputs");
    let mut shared_ref = vec![0.0f32; NUM_TOKENS * HIDDEN];
    for t in 0..NUM_TOKENS {
        for j in 0..HIDDEN {
            let mut acc = 0.0f32;
            for kk in 0..K_INTERMEDIATE {
                acc += a[t * K_INTERMEDIATE + kk] * w[kk * HIDDEN + j];
            }
            shared_ref[t * HIDDEN + j] = acc;
        }
    }
    assert_buf_close(&shared_out, &shared_ref, 1e-5, 1e-4);
    assert!(shared_out.iter().all(|v| v.is_finite()));

    // Build `[NUM_TOKENS, TOP_K, HIDDEN]` expert_outs and per-token
    // top-k weights, then reduce.
    let expert_outs = build_expert_outs(&a, &b, &plan, &picks_per_token);
    let weights = build_weights(&picks_per_token);
    let mut reduced = vec![0.0f32; NUM_TOKENS * HIDDEN];
    weighted_reduce_tiled(&expert_outs, &weights, TOP_K, HIDDEN, &mut reduced)
        .expect("weighted_reduce_tiled must accept well-formed inputs");
    let mut reduced_ref = vec![0.0f32; NUM_TOKENS * HIDDEN];
    for t in 0..NUM_TOKENS {
        for j in 0..HIDDEN {
            let mut acc = 0.0f32;
            for e_idx in 0..TOP_K {
                let w_e = weights[t * TOP_K + e_idx];
                let v = expert_outs[(t * TOP_K + e_idx) * HIDDEN + j];
                acc += w_e * v;
            }
            reduced_ref[t * HIDDEN + j] = acc;
        }
    }
    assert_buf_close(&reduced, &reduced_ref, 1e-5, 1e-4);
    assert!(reduced.iter().all(|v| v.is_finite()));

    // Writeback: stage the routed-expert outputs (the first top-k
    // pick per token, which is what the dispatch plan routed) and
    // coalesce-write into the residual buffer.
    let mut routed = vec![0.0f32; NUM_TOKENS * HIDDEN];
    grouped_gemm_tiled(
        &a,
        &b,
        &plan.expert_buckets,
        0,
        K_INTERMEDIATE,
        HIDDEN,
        &mut routed,
    )
    .expect("grouped_gemm_tiled must accept well-formed inputs");
    let stage = stage_expert_outputs(&routed, &plan, HIDDEN)
        .expect("stage_expert_outputs must accept well-formed inputs");
    let mut writeback = vec![0.0f32; NUM_TOKENS * HIDDEN];
    coalesced_writeback(&stage, NUM_TOKENS, HIDDEN, &mut writeback)
        .expect("coalesced_writeback must accept well-formed inputs");

    // Hand-rolled residual reference: per-(token, expert) copy of the
    // routed row. For dropped tokens the row is zero.
    let mut writeback_ref = [0.0f32; NUM_TOKENS * HIDDEN];
    for t in 0..NUM_TOKENS {
        if plan.dropped.contains(&t) {
            continue;
        }
        for j in 0..HIDDEN {
            writeback_ref[t * HIDDEN + j] = routed[t * HIDDEN + j];
        }
    }
    for (i, (&x, &y)) in writeback.iter().zip(writeback_ref.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "writeback byte-eq drift at {i}: got {x}, expected {y}"
        );
    }
    assert!(writeback.iter().all(|v| v.is_finite()));

    // Per-token top-k weights must be finite-positive.
    for (t, picks) in picks_per_token.iter().enumerate() {
        for (e, w_pick) in picks {
            assert!(
                w_pick.is_finite() && *w_pick > 0.0,
                "token {t} pick {e} weight not finite/positive"
            );
        }
    }
}

/// (b) `grouped_gemm_tiled` must produce byte-equal output to the
/// scalar `grouped_gemm` on the same dispatch plan.
#[test]
fn qwen_moe_v2_grouped_gemm_tiled_matches_scalar_grouped_gemm() {
    let router_logits = build_router_logits();
    let (assignments, _picks_per_token) = run_router(&router_logits);

    let token_indices: Vec<usize> = (0..NUM_TOKENS).collect();
    let plan: DispatchPlan = moe_dispatch(
        &token_indices,
        &assignments,
        NUM_EXPERTS,
        CAPACITY_FACTOR,
    )
    .expect("dispatch must accept well-formed inputs");

    let a = deterministic_vec(NUM_TOKENS * K_INTERMEDIATE, 0xA_CE);
    let b: Vec<f32> = (0..NUM_EXPERTS)
        .flat_map(|e| deterministic_vec(K_INTERMEDIATE * HIDDEN, 0xB0_E0 + e as u64))
        .collect();

    let mut scalar_out = vec![0.0f32; NUM_TOKENS * HIDDEN];
    grouped_gemm(
        &a,
        &b,
        &plan.expert_buckets,
        0,
        K_INTERMEDIATE,
        HIDDEN,
        &mut scalar_out,
    )
    .expect("scalar grouped_gemm must accept well-formed inputs");

    let mut tiled_out = vec![0.0f32; NUM_TOKENS * HIDDEN];
    grouped_gemm_tiled(
        &a,
        &b,
        &plan.expert_buckets,
        0,
        K_INTERMEDIATE,
        HIDDEN,
        &mut tiled_out,
    )
    .expect("tiled grouped_gemm must accept well-formed inputs");

    assert_eq!(scalar_out.len(), tiled_out.len());
    for (i, (&x, &y)) in scalar_out.iter().zip(tiled_out.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "grouped_gemm tiled-vs-scalar byte-eq drift at {i}: tiled={x} scalar={y}"
        );
    }
}

/// (c) `weighted_reduce_tiled` must produce byte-equal output to the
/// scalar `weighted_reduce` on the same `(expert_outs, weights)`.
#[test]
fn qwen_moe_v2_weighted_reduce_tiled_matches_scalar_weighted_reduce() {
    let router_logits = build_router_logits();
    let (assignments, picks_per_token) = run_router(&router_logits);

    let token_indices: Vec<usize> = (0..NUM_TOKENS).collect();
    let plan: DispatchPlan = moe_dispatch(
        &token_indices,
        &assignments,
        NUM_EXPERTS,
        CAPACITY_FACTOR,
    )
    .expect("dispatch must accept well-formed inputs");

    let a = deterministic_vec(NUM_TOKENS * K_INTERMEDIATE, 0xA_CE);
    let b: Vec<f32> = (0..NUM_EXPERTS)
        .flat_map(|e| deterministic_vec(K_INTERMEDIATE * HIDDEN, 0xB0_E0 + e as u64))
        .collect();

    let expert_outs = build_expert_outs(&a, &b, &plan, &picks_per_token);
    let weights = build_weights(&picks_per_token);

    let mut scalar_out = vec![0.0f32; NUM_TOKENS * HIDDEN];
    weighted_reduce(&expert_outs, &weights, TOP_K, HIDDEN, &mut scalar_out)
        .expect("scalar weighted_reduce must accept well-formed inputs");

    let mut tiled_out = vec![0.0f32; NUM_TOKENS * HIDDEN];
    weighted_reduce_tiled(&expert_outs, &weights, TOP_K, HIDDEN, &mut tiled_out)
        .expect("tiled weighted_reduce must accept well-formed inputs");

    assert_eq!(scalar_out.len(), tiled_out.len());
    for (i, (&x, &y)) in scalar_out.iter().zip(tiled_out.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "weighted_reduce tiled-vs-scalar byte-eq drift at {i}: tiled={x} scalar={y}"
        );
    }
}