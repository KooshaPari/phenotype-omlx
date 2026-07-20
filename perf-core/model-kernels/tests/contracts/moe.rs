//! Section "MoE" of the original contracts.rs.
//!
//! Split out of the original monolithic `model-kernels/tests/contracts.rs`
//! (1130 lines) so each topic stays under the 350-line target. Test bodies
//! are byte-identical to the source file; only the surrounding module
//! wrapper and `use super::*;` import differ.

use super::*;


#[test]
fn router_topk_returns_k_distinct_experts() {
    let num_experts = 8;
    let top_k = 2;
    let logits = vec![3.0, 1.0, 2.0, 0.5, 4.0, 0.0, 1.5, 0.75];
    let picks = router_topk(&logits, num_experts, top_k, 0).unwrap();
    assert_eq!(picks.len(), top_k);
    // Distinct experts.
    let unique: std::collections::HashSet<usize> = picks.iter().map(|(e, _)| *e).collect();
    assert_eq!(unique.len(), top_k);
    // Top two by logit are 4 (logit 4.0) and 0 (logit 3.0).
    assert!(picks.iter().any(|(e, _)| *e == 4));
    assert!(picks.iter().any(|(e, _)| *e == 0));
}

#[test]
fn router_topk_is_stable_under_seed_replay() {
    let num_experts = 16;
    let top_k = 4;
    let logits = deterministic_vec(num_experts, 31);
    let picks1 = router_topk(&logits, num_experts, top_k, 99).unwrap();
    let picks2 = router_topk(&logits, num_experts, top_k, 99).unwrap();
    assert_eq!(picks1, picks2);
}

#[test]
fn router_topk_sums_weights_to_at_most_one_after_renormalisation() {
    let num_experts = 6;
    let top_k = 3;
    let logits = vec![2.0, 1.0, 0.5, 0.0, 1.5, 0.75];
    let picks = router_topk(&logits, num_experts, top_k, 0).unwrap();
    let total: f32 = picks.iter().map(|(_, w)| *w).sum();
    assert!(total <= 1.0 + 1e-6, "sum {total} > 1.0");
    assert!((total - 1.0).abs() < 1e-5, "renormalized sum should be ~1, got {total}");
    for (_, w) in &picks {
        assert!(*w > 0.0);
    }
}

#[test]
fn moe_dispatch_respects_capacity_factor() {
    let num_experts = 4;
    let num_tokens = 8;
    let capacity_factor = 1.0;
    let assignments: Vec<(usize, f32)> = vec![
        (0, 0.9),
        (1, 0.8),
        (0, 0.7),
        (2, 0.6),
        (1, 0.5),
        (3, 0.4),
        (2, 0.3),
        (3, 0.2),
    ];
    let plan = moe_dispatch(
        &(0..num_tokens).collect::<Vec<_>>(),
        &assignments,
        num_experts,
        capacity_factor,
    )
    .unwrap();
    // capacity = ceil(1.0 * 8 / 4) = 2 per expert.
    for (i, used) in plan.capacity_used.iter().enumerate() {
        assert!(*used <= 2, "expert {i} used {used} > 2");
    }
    assert_eq!(plan.expert_buckets.len(), num_experts);
}

#[test]
fn moe_dispatch_drops_excess_tokens_into_dropped_bucket() {
    let num_experts = 2;
    let capacity_factor = 1.0; // capacity per expert = ceil(1.0 * 6 / 2) = 3
    let assignments: Vec<(usize, f32)> = vec![
        (0, 0.9),
        (0, 0.8),
        (0, 0.7),
        (0, 0.6), // 4 tokens to expert 0 -> 1 dropped
        (1, 0.5),
        (1, 0.4),
    ];
    let plan = moe_dispatch(
        &(0..assignments.len()).collect::<Vec<_>>(),
        &assignments,
        num_experts,
        capacity_factor,
    )
    .unwrap();
    assert_eq!(plan.capacity_used[0], 3);
    assert_eq!(plan.capacity_used[1], 2);
    assert_eq!(plan.dropped.len(), 1, "expected 1 dropped, got {:?}", plan.dropped);
}

#[test]
fn grouped_gemm_matches_per_bucket_scalar_oracle() {
    // Two experts, each handles two tokens. m=2 (rows per bucket),
    // k=3, n=2.
    let m = 2;
    let k = 3;
    let n = 2;
    // a is laid out [num_tokens, k] with row order = token order.
    let a = vec![
        1.0, 0.0, 0.5, // token 0
        0.0, 1.0, 0.5, // token 1
        1.0, 1.0, 0.0, // token 2
        0.5, 0.5, 1.0, // token 3
    ];
    let b = vec![
        // expert 0 weight (k x n)
        1.0, 0.0, //
        0.0, 1.0, //
        0.5, 0.5, //
        // expert 1 weight
        0.0, 1.0, //
        1.0, 0.0, //
        1.0, 1.0, //
    ];
    let buckets = vec![vec![0usize, 1], vec![2, 3]];
    let mut out = vec![0.0f32; a.len() / k * n];
    grouped_gemm(&a, &b, &buckets, m, k, n, &mut out).unwrap();

    // Manual oracle: for each bucket, compute the dense matmul on the
    // rows that bucket selects.
    for (e, bucket) in buckets.iter().enumerate() {
        let b_offset = e * k * n;
        for (&tok, _row_idx) in bucket.iter().zip(0..) {
            let mut row = vec![0.0f32; n];
            for j in 0..n {
                for kk in 0..k {
                    row[j] += a[tok * k + kk] * b[b_offset + kk * n + j];
                }
            }
            let dst_base = tok * n;
            assert_buf_close(&out[dst_base..dst_base + n], &row, 1e-5, 1e-4);
        }
    }
}

#[test]
fn weighted_reduce_matches_per_token_scalar_oracle() {
    // Two experts per token, two tokens.
    let n = 2; // hidden dim
    let tokens = 2;
    let experts_per_token = 2;
    // expert_outs: laid out per-token-major: each token has
    // `experts_per_token` rows of size `n`.
    let expert_outs = vec![
        // token 0
        1.0, 0.0, // expert A
        0.0, 1.0, // expert B
        // token 1
        2.0, 2.0, // expert A
        1.0, -1.0, // expert B
    ];
    let weights = vec![
        0.6, 0.4, // token 0
        0.7, 0.3, // token 1
    ];
    let mut out = vec![0.0f32; tokens * n];
    weighted_reduce(&expert_outs, &weights, experts_per_token, n, &mut out).unwrap();
    let expected = vec![0.6 * 1.0 + 0.4 * 0.0, 0.6 * 0.0 + 0.4 * 1.0, 0.7 * 2.0 + 0.3 * 1.0, 0.7 * 2.0 - 0.3 * 1.0];
    assert_buf_close(&out, &expected, 1e-5, 1e-4);
}

#[test]
fn shared_expert_matches_dense_matmul() {
    let n = 2;
    let k = 3;
    let tokens = 2;
    let x = vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5];
    let w = vec![
        1.0, 0.0, //
        0.0, 1.0, //
        0.5, 0.5, //
    ];
    let mut out = vec![0.0f32; tokens * n];
    shared_expert(&x, &w, &mut out).unwrap();
    // Reference: per token, x[t] @ w.
    let mut expected = vec![0.0f32; tokens * n];
    for t in 0..tokens {
        for j in 0..n {
            for kk in 0..k {
                expected[t * n + j] += x[t * k + kk] * w[kk * n + j];
            }
        }
    }
    assert_buf_close(&out, &expected, 1e-5, 1e-4);
}

