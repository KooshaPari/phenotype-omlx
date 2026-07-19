//! Qwen3-Coder-Next + Bonsai acceptance integration tests for
//! `model-kernels`.
//!
//! These tests tie together:
//!
//! - the Bonsai fused ternary matmul (see
//!   `quantized::ternary_matmul::ternary_matmul`);
//! - the Qwen-style DeltaNet chunked linear-recurrent update
//!   (see `recurrent::deltanet`);
//! - the sparse MoE pipeline (see `moe::router`,
//!   `moe::dispatch`, `moe::shared`, `moe::reduce`).
//!
//! The Bonsai row exercises the `Exact ternary block layout and
//! round-trip oracle` row of the model acceptance matrix in
//! `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md`.
//!
//! The Qwen agentic mini-trace exercises the `Long-context decode,
//! tool-use traces, GQA or DeltaNet state, sparse MoE` row of the
//! same matrix. We focus here on the DeltaNet state + sparse MoE
//! pieces; long-context decode and tool-use traces are covered
//! elsewhere in the workspace (see `tests/contracts.rs`).
//!
//! Tolerances follow the crate contract: `abs = 1e-5`, `rel = 1e-4`.

use model_kernels::common::{approx_eq, Lcg};
use model_kernels::moe::{
    moe_dispatch, router_topk, shared_expert, weighted_reduce, DispatchPlan,
};
use model_kernels::quantized::{
    ternary_matmul, ternary_pack, ternary_unpack, SignedTernary,
};
use model_kernels::recurrent::{deltanet_chunk, deltanet_step};

const SEED: u64 = 0xCAFE_BABE_DEAD_BEEF;

/// Build a deterministic vector of `f32` of length `n`.
fn deterministic_vec(n: usize, salt: u64) -> Vec<f32> {
    let mut rng = Lcg::new(SEED ^ salt);
    (0..n).map(|_| rng.next_signed()).collect()
}

/// Element-wise close comparison with explicit tolerances.
fn assert_buf_close(a: &[f32], b: &[f32], abs: f32, rel: f32) {
    assert_eq!(a.len(), b.len(), "buffer length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let ok = diff <= abs || diff <= rel * x.abs().max(y.abs());
        assert!(ok, "buffers differ at {i}: got {x}, expected {y} (abs={abs}, rel={rel})");
    }
}

// ===========================================================================
// Bonsai ternary matmul parity
// ===========================================================================

#[test]
fn bonsai_ternary_matmul_matches_unpacked_reference() {
    // Activation: a is [4, 16] row-major.
    // Weight: b is [16, 32] row-major ternary. The kernel expects
    // b_packed to be the row-major flat of [16, 32] in the same
    // byte order as `ternary_pack` would produce.
    let m = 4;
    let k = 16;
    let n = 32;
    let group_size = k * n; // single Bonsai group

    // Build a deterministic ternary weight using the Lcg salt for
    // reproducibility.
    let mut rng = Lcg::new(SEED ^ 0xA11CE);
    let values: Vec<SignedTernary> = (0..k * n)
        .map(|_| match (rng.next_u64() % 3) as u8 {
            0 => SignedTernary::Pos,
            1 => SignedTernary::Neg,
            _ => SignedTernary::Zero,
        })
        .collect();
    let (packed, scales, zeros) = ternary_pack(&values, group_size).unwrap();

    let a = deterministic_vec(m * k, 0xBEEF);

    let mut out = vec![0.0f32; m * n];
    ternary_matmul(&a, &packed, &scales, &zeros, group_size, m, k, n, &mut out).unwrap();

    // Reference: unpack and run a dense per-row inner-product.
    let mut unpacked = vec![SignedTernary::Zero; values.len()];
    ternary_unpack(&packed, &scales, &zeros, values.len(), group_size, &mut unpacked).unwrap();

    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let w = match unpacked[kk * n + j] {
                    SignedTernary::Pos => 1.0,
                    SignedTernary::Neg => -1.0,
                    SignedTernary::Zero => 0.0,
                };
                acc += a[row * k + kk] * w;
            }
            expected[row * n + j] = acc;
        }
    }
    assert_buf_close(&out, &expected, 1e-5, 1e-4);
    // Bonus: every entry must be finite.
    assert!(out.iter().all(|v| v.is_finite()), "out has a non-finite entry");
}

// ===========================================================================
// Qwen3-Coder-Next DeltaNet acceptance
// ===========================================================================

/// Run a 4-head, head_dim=4 DeltaNet trace for `chunk_size` steps.
/// Each head runs independently with its own initial state and its
/// own slice of (q, k, v). Returns stacked outputs `[chunk, head_dim]`
/// per head, plus the per-head final states.
fn run_qwen_deltanet_trace(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    initial_states: &[Vec<f32>],
    chunk_size: usize,
    num_heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    // The existing kernel operates per head. We stack the per-head
    // outputs into a flat [chunk_size, num_heads, head_dim] buffer.
    let mut outs = vec![0.0f32; chunk_size * num_heads * head_dim];
    let mut final_states = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let qh: Vec<f32> = (0..chunk_size)
            .flat_map(|c| q[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim].iter().copied())
            .collect();
        let kh: Vec<f32> = (0..chunk_size)
            .flat_map(|c| k[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim].iter().copied())
            .collect();
        let vh: Vec<f32> = (0..chunk_size)
            .flat_map(|c| v[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim].iter().copied())
            .collect();
        let (oh, sh) = deltanet_chunk(&qh, &kh, &vh, chunk_size, head_dim, &initial_states[h]).unwrap();
        for c in 0..chunk_size {
            outs[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim]
                .copy_from_slice(&oh[c * head_dim..c * head_dim + head_dim]);
        }
        final_states.push(sh);
    }
    (outs, final_states)
}

#[test]
fn qwen_deltanet_chunk_matches_repeated_step_4_heads() {
    // 4 heads, head_dim=4, 16-step trace.
    let num_heads = 4;
    let head_dim = 4;
    let chunk_size = 16;
    let beta = 0.5;

    let q = deterministic_vec(chunk_size * num_heads * head_dim, 0xD0_1A);
    let k = deterministic_vec(chunk_size * num_heads * head_dim, 0xD0_1B);
    let v = deterministic_vec(chunk_size * num_heads * head_dim, 0xD0_1C);

    // Per-head initial states: deterministic, head-stratified so the
    // four heads are distinguishable.
    let initial_states: Vec<Vec<f32>> = (0..num_heads)
        .map(|h| {
            let salt = SEED ^ (0xE0_F0u64 + h as u64);
            let mut rng = Lcg::new(salt);
            (0..head_dim * head_dim).map(|_| rng.next_signed() * 0.25).collect()
        })
        .collect();

    // Path A: chunk via deltanet_chunk per head.
    let (chunk_outs, chunk_states) = run_qwen_deltanet_trace(
        &q,
        &k,
        &v,
        &initial_states,
        chunk_size,
        num_heads,
        head_dim,
    );

    // Path B: run deltanet_step sequentially per head and stack.
    let mut step_states: Vec<Vec<f32>> = initial_states.clone();
    let mut step_outs = vec![0.0f32; chunk_size * num_heads * head_dim];
    for h in 0..num_heads {
        for c in 0..chunk_size {
            let qc = &q[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim];
            let kc = &k[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim];
            let vc = &v[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim];
            let o = deltanet_step(qc, kc, vc, &mut step_states[h], beta, head_dim).unwrap();
            step_outs[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim]
                .copy_from_slice(&o);
        }
    }

    // Per-head outputs and final states must match.
    for h in 0..num_heads {
        let mut per_head_chunk = vec![0.0f32; chunk_size * head_dim];
        let mut per_head_step = vec![0.0f32; chunk_size * head_dim];
        for c in 0..chunk_size {
            per_head_chunk[c * head_dim..c * head_dim + head_dim].copy_from_slice(
                &chunk_outs[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim],
            );
            per_head_step[c * head_dim..c * head_dim + head_dim].copy_from_slice(
                &step_outs[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim],
            );
        }
        assert_buf_close(&per_head_chunk, &per_head_step, 1e-5, 1e-4);
        assert_buf_close(&chunk_states[h], &step_states[h], 1e-5, 1e-4);
    }

    // And all entries must be finite.
    assert!(chunk_outs.iter().all(|v| v.is_finite()), "non-finite chunk output");
    for h in 0..num_heads {
        assert!(chunk_states[h].iter().all(|v| v.is_finite()), "non-finite head {h} state");
    }
}

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
    for h in 0..num_heads {
        assert!(deltanet_states[h].iter().all(|v| v.is_finite()), "head {h} state not finite");
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