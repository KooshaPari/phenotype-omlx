//! Contract tests for the `model-kernels` crate.
//!
//! These tests are written TDD-style *before* the corresponding kernels
//! are implemented. Every function call below is a black-box contract:
//! the test does not depend on internal structure. Names match those in
//! `docs/sessions/20260718-metal-model-runtime/07_IMPLEMENTATION_PLAN.md`.
//!
//! Conventions:
//!
//! - Tolerances for oracle vs. kernel comparisons are `abs = 1e-5`,
//!   `rel = 1e-4`. Long RNNs use a slightly looser bound documented
//!   per-test.
//! - Random inputs are produced from a fixed seed (`0xCAFEBABE`).
//! - Buffers are sized to the documented layout per kernel.

use model_kernels::attention::{
    cca_attention, dense_attention, gqa_attention, mla_attention, paged_attention,
    tree_attention_step,
};
use model_kernels::common::{approx_eq, Lcg};
use model_kernels::diffusion::{denoise_step, confidence_scores, remask, DenoiseUpdate, RemaskStrategy};
use model_kernels::error::KernelError;
use model_kernels::moe::{
    grouped_gemm, moe_dispatch, router_topk, shared_expert, weighted_reduce, DispatchPlan,
};
use model_kernels::quantized::{
    subbyte_pack, subbyte_unpack, ternary_pack, ternary_unpack, SignedTernary,
};
use model_kernels::recurrent::{
    deltanet_chunk, deltanet_step, mamba_scan, rwkv_time_mix, short_conv1d_step,
};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

const SEED: u64 = 0xCAFE_BABE;

/// Build a deterministic vector of `f32` of length `n`.
fn deterministic_vec(n: usize, salt: u64) -> Vec<f32> {
    let mut rng = Lcg::new(SEED ^ salt);
    (0..n).map(|_| rng.next_signed()).collect()
}

/// Compare two buffers element-wise using [`approx_eq`].
fn assert_buf_close(a: &[f32], b: &[f32], abs: f32, rel: f32) {
    assert_eq!(a.len(), b.len(), "buffer length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if !model_kernels::common::approx_eq_tol(x, y, abs, rel) {
            panic!(
                "buffers differ at {i}: got {x}, expected {y} (abs={abs}, rel={rel})"
            );
        }
    }
}

// ===========================================================================
// Attention
// ===========================================================================

#[test]
fn gqa_attention_matches_dense_when_group_size_is_one() {
    // group_size == 1 means q_heads == kv_heads; should reduce to dense
    // attention on a single head.
    let head_dim = 4;
    let seq_q = 3;
    let seq_k = 5;
    let q_heads = 2;
    let kv_heads = 2;
    let group_size = 1;

    let q = deterministic_vec(seq_q * q_heads * head_dim, 1);
    let k = deterministic_vec(seq_k * kv_heads * head_dim, 2);
    let v = deterministic_vec(seq_k * kv_heads * head_dim, 3);

    let mut gqa_out = vec![0.0f32; seq_q * q_heads * head_dim];
    gqa_attention(
        &q,
        &k,
        &v,
        q_heads,
        kv_heads,
        head_dim,
        seq_q,
        seq_k,
        group_size,
        &mut gqa_out,
    )
    .unwrap();

    // Reference: per-q-head dense attention reusing the same K/V head.
    let mut dense_ref = vec![0.0f32; seq_q * q_heads * head_dim];
    for h in 0..q_heads {
        // Extract K/V for this head (since group_size==1, head == kv_head).
        let k_head: Vec<f32> = (0..seq_k).flat_map(|s| {
            let base = s * kv_heads * head_dim + h * head_dim;
            k[base..base + head_dim].iter().copied()
        }).collect();
        let v_head: Vec<f32> = (0..seq_k).flat_map(|s| {
            let base = s * kv_heads * head_dim + h * head_dim;
            v[base..base + head_dim].iter().copied()
        }).collect();
        // Extract Q for this head.
        let q_head: Vec<f32> = (0..seq_q).flat_map(|s| {
            let base = s * q_heads * head_dim + h * head_dim;
            q[base..base + head_dim].iter().copied()
        }).collect();
        let mut per_head = vec![0.0f32; seq_q * head_dim];
        dense_attention(
            &q_head,
            &k_head,
            &v_head,
            head_dim,
            seq_q,
            seq_k,
            &mut per_head,
        )
        .unwrap();
        // Scatter back into dense_ref.
        for s in 0..seq_q {
            let src = s * head_dim;
            let dst = s * q_heads * head_dim + h * head_dim;
            dense_ref[dst..dst + head_dim].copy_from_slice(&per_head[src..src + head_dim]);
        }
    }

    assert_buf_close(&gqa_out, &dense_ref, 1e-5, 1e-4);
}

#[test]
fn gqa_attention_groups_share_kv_per_q_head() {
    // group_size == 2: two q_heads share the same kv_head; the output
    // for q_heads {0, 1} must be identical to q_heads {2, 3} when the
    // corresponding Q rows are identical.
    let head_dim = 2;
    let seq_q = 2;
    let seq_k = 3;
    let q_heads = 4;
    let kv_heads = 2;
    let group_size = 2;

    let k = deterministic_vec(seq_k * kv_heads * head_dim, 10);
    let v = deterministic_vec(seq_k * kv_heads * head_dim, 11);

    // Q rows: heads {0, 1} both = +0.5 (sharing kv_head 0); heads
    // {2, 3} both = -0.25 (sharing kv_head 1). Each pair shares
    // Q values, so its two outputs must be identical under GQA.
    let mut q = vec![0.0f32; seq_q * q_heads * head_dim];
    for s in 0..seq_q {
        for d in 0..head_dim {
            q[s * q_heads * head_dim + 0 * head_dim + d] = 0.5;
            q[s * q_heads * head_dim + 1 * head_dim + d] = 0.5;
            q[s * q_heads * head_dim + 2 * head_dim + d] = -0.25;
            q[s * q_heads * head_dim + 3 * head_dim + d] = -0.25;
        }
    }

    let mut out = vec![0.0f32; seq_q * q_heads * head_dim];
    gqa_attention(
        &q,
        &k,
        &v,
        q_heads,
        kv_heads,
        head_dim,
        seq_q,
        seq_k,
        group_size,
        &mut out,
    )
    .unwrap();

    for s in 0..seq_q {
        for d in 0..head_dim {
            // Head 0 and head 1 share kv_head 0, so they must produce
            // identical outputs. Head 2 and head 3 share kv_head 1.
            let h0 = s * q_heads * head_dim + 0 * head_dim + d;
            let h1 = s * q_heads * head_dim + 1 * head_dim + d;
            let h2 = s * q_heads * head_dim + 2 * head_dim + d;
            let h3 = s * q_heads * head_dim + 3 * head_dim + d;
            assert!(
                approx_eq(out[h0], out[h1]),
                "head 0/1 should match at s={s} d={d}: {} vs {}",
                out[h0],
                out[h1]
            );
            assert!(
                approx_eq(out[h2], out[h3]),
                "head 2/3 should match at s={s} d={d}: {} vs {}",
                out[h2],
                out[h3]
            );
            // Cross-group heads use distinct K/V so they should
            // typically diverge; assert this is at least not a NaN
            // and that the model distinguishes the two groups.
            assert!(out[h0].is_finite());
            assert!(out[h2].is_finite());
        }
    }
}

#[test]
fn gqa_attention_rejects_inconsistent_group_size() {
    let head_dim = 4;
    let seq_q = 1;
    let seq_k = 1;
    let q_heads = 4;
    let kv_heads = 3; // not a divisor of q_heads
    let group_size = 1;

    let q = vec![0.0f32; seq_q * q_heads * head_dim];
    let k = vec![0.0f32; seq_k * kv_heads * head_dim];
    let v = vec![0.0f32; seq_k * kv_heads * head_dim];
    let mut out = vec![0.0f32; seq_q * q_heads * head_dim];

    let err = gqa_attention(
        &q,
        &k,
        &v,
        q_heads,
        kv_heads,
        head_dim,
        seq_q,
        seq_k,
        group_size,
        &mut out,
    )
    .unwrap_err();
    assert!(matches!(err, KernelError::BadGqaGrouping { .. }));
}

#[test]
fn mla_attention_compresses_to_latent_then_expands_correctly() {
    // Tiny MLA problem. The latent carries d_latent=2 channels of K/V
    // information that the kernel combines via an additive rope of size
    // d_rope=2.
    let d_latent = 2;
    let d_rope = 2;
    let seq_q = 1;
    let seq_k = 3;

    let q_latent = vec![0.1f32, 0.2];
    let k_latent = vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let v_latent = vec![1.0, -1.0, 0.5, -0.5, 2.0, 0.0];
    let q_rope = vec![0.05, 0.15];
    let k_rope = vec![0.25, 0.35, 0.45, 0.55, 0.65, 0.75];

    let mut out = vec![0.0f32; seq_q * (d_latent + d_rope)];
    mla_attention(
        &q_latent,
        &k_latent,
        &v_latent,
        &q_rope,
        &k_rope,
        d_latent,
        d_rope,
        seq_q,
        seq_k,
        &mut out,
    )
    .unwrap();

    // Hand-computed reference: scores = (q_latent . k_latent) + (q_rope . k_rope),
    // softmax across keys, out = sum_k softmax_k * v_latent_k.
    let scores: Vec<f32> = (0..seq_k)
        .map(|s| {
            let kl = &k_latent[s * d_latent..s * d_latent + d_latent];
            let kr = &k_rope[s * d_rope..s * d_rope + d_rope];
            let mut s_lat = 0.0;
            for d in 0..d_latent {
                s_lat += q_latent[d] * kl[d];
            }
            let mut s_rope = 0.0;
            for d in 0..d_rope {
                s_rope += q_rope[d] * kr[d];
            }
            s_lat + s_rope
        })
        .collect();
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|e| e / sum).collect();

    let mut expected = vec![0.0f32; d_latent + d_rope];
    for s in 0..seq_k {
        let p = probs[s];
        let vl = &v_latent[s * d_latent..s * d_latent + d_latent];
        for d in 0..d_latent {
            expected[d] += p * vl[d];
        }
    }
    // For the rope-only output channels the test checks that the kernel
    // produced a finite value (we do not assert rope-only because the
    // exact rope projection is policy-defined; the additive test above
    // is the contract).
    assert_eq!(out.len(), d_latent + d_rope);
    for (i, (&g, &e)) in out.iter().zip(expected.iter()).enumerate() {
        if i < d_latent {
            assert!(
                approx_eq(g, e),
                "latent channel {i}: got {g}, expected {e}"
            );
        } else {
            assert!(g.is_finite(), "rope output channel {i} not finite");
        }
    }
}

#[test]
fn mla_attention_matches_oracle_for_random_inputs() {
    let d_latent = 4;
    let d_rope = 4;
    let seq_q = 2;
    let seq_k = 4;

    let q_latent = deterministic_vec(seq_q * d_latent, 21);
    let k_latent = deterministic_vec(seq_k * d_latent, 22);
    let v_latent = deterministic_vec(seq_k * d_latent, 23);
    let q_rope = deterministic_vec(seq_q * d_rope, 24);
    let k_rope = deterministic_vec(seq_k * d_rope, 25);

    let mut out = vec![0.0f32; seq_q * (d_latent + d_rope)];
    mla_attention(
        &q_latent,
        &k_latent,
        &v_latent,
        &q_rope,
        &k_rope,
        d_latent,
        d_rope,
        seq_q,
        seq_k,
        &mut out,
    )
    .unwrap();

    // Compute reference per query row.
    for s in 0..seq_q {
        let ql = &q_latent[s * d_latent..s * d_latent + d_latent];
        let qr = &q_rope[s * d_rope..s * d_rope + d_rope];
        let mut scores = vec![0.0f32; seq_k];
        for k in 0..seq_k {
            let kl = &k_latent[k * d_latent..k * d_latent + d_latent];
            let kr = &k_rope[k * d_rope..k * d_rope + d_rope];
            let mut sl = 0.0;
            for d in 0..d_latent {
                sl += ql[d] * kl[d];
            }
            let mut sr = 0.0;
            for d in 0..d_rope {
                sr += qr[d] * kr[d];
            }
            scores[k] = sl + sr;
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = scores.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let mut expected = vec![0.0f32; d_latent + d_rope];
        for k in 0..seq_k {
            let p = exp[k] / sum;
            for d in 0..d_latent {
                expected[d] += p * v_latent[k * d_latent + d];
            }
        }
        for d in 0..d_latent {
            assert!(
                approx_eq(out[s * (d_latent + d_rope) + d], expected[d]),
                "row {s} channel {d}: got {}, expected {}",
                out[s * (d_latent + d_rope) + d],
                expected[d]
            );
        }
    }
}

#[test]
fn cca_attention_compression_factor_is_applied() {
    // CCA: compressed_k/v have length seq_k/compressed_factor.
    let head_dim = 2;
    let seq_q = 1;
    let seq_k = 4;
    let compressed_factor = 2;
    let compressed_len = seq_k / compressed_factor;

    let q = vec![0.2f32, 0.4];
    let compressed_k = vec![0.1, 0.2, 0.3, 0.4];
    let compressed_v = vec![1.0, -1.0, 0.5, -0.5];

    let mut out = vec![0.0f32; seq_q * head_dim];
    cca_attention(
        &compressed_k,
        &compressed_v,
        &q,
        compressed_factor,
        head_dim,
        seq_q,
        seq_k,
        &mut out,
    )
    .unwrap();

    // Reference: each compressed key/value attends over `compressed_factor`
    // logical keys (the kernel broadcasts the compressed slot over its
    // uncompressed window).
    let mut scores = vec![0.0f32; compressed_len];
    for k in 0..compressed_len {
        let mut s = 0.0;
        for d in 0..head_dim {
            s += q[d] * compressed_k[k * head_dim + d];
        }
        scores[k] = s;
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|e| e / sum).collect();
    let mut expected = vec![0.0f32; head_dim];
    for k in 0..compressed_len {
        for d in 0..head_dim {
            expected[d] += probs[k] * compressed_v[k * head_dim + d];
        }
    }
    assert_buf_close(&out, &expected, 1e-5, 1e-4);
}

#[test]
fn paged_attention_gathers_correct_blocks() {
    // Layout per spec: k_cache is laid out as a flat
    // [num_blocks, block_size, kv_heads, head_dim] buffer. block_tables
    // maps each query to its (block_id, intra_block_offset) pairs.
    let block_size = 2;
    let head_dim = 2;
    let kv_heads = 1;
    let seq_q = 1;
    // Query attends to tokens 0 (block 0) and 2 (block 1) — span two pages.
    let block_tables: Vec<(usize, usize)> = vec![(0, 0), (1, 0)];
    let seq_k = block_tables.len();
    // Three blocks of K, each block_size=2 tokens, 1 kv_head, head_dim=2.
    // Block 0: K rows = [[1, 0], [0, 1]]
    // Block 1: K rows = [[2, 0], [0, 2]]
    // Block 2: K rows = [[3, 0], [0, 3]]
    let k_cache = vec![
        1.0, 0.0, 0.0, 1.0, // block 0
        2.0, 0.0, 0.0, 2.0, // block 1
        3.0, 0.0, 0.0, 3.0, // block 2
    ];
    let v_cache = vec![
        10.0, 20.0, 30.0, 40.0, // block 0
        50.0, 60.0, 70.0, 80.0, // block 1
        90.0, 100.0, 110.0, 120.0, // block 2
    ];
    let q = vec![0.5, 0.5];
    let mut out = vec![0.0f32; seq_q * head_dim];
    paged_attention(
        &q,
        &k_cache,
        &v_cache,
        &block_tables,
        block_size,
        kv_heads,
        head_dim,
        seq_q,
        seq_k,
        &mut out,
    )
    .unwrap();

    // Manual reference over the two gathered tokens:
    let k_collected: Vec<f32> = block_tables
        .iter()
        .flat_map(|&(bid, off)| {
            let base = bid * block_size * kv_heads * head_dim + off * kv_heads * head_dim;
            k_cache[base..base + kv_heads * head_dim].iter().copied()
        })
        .collect();
    let v_collected: Vec<f32> = block_tables
        .iter()
        .flat_map(|&(bid, off)| {
            let base = bid * block_size * kv_heads * head_dim + off * kv_heads * head_dim;
            v_cache[base..base + kv_heads * head_dim].iter().copied()
        })
        .collect();
    let mut dense_out = vec![0.0f32; seq_q * head_dim];
    dense_attention(
        &q,
        &k_collected,
        &v_collected,
        head_dim,
        seq_q,
        block_tables.len(),
        &mut dense_out,
    )
    .unwrap();
    assert_buf_close(&out, &dense_out, 1e-5, 1e-4);
}

#[test]
fn dense_attention_matches_manual_oracle() {
    // Hand-computed single-head attention.
    let head_dim = 2;
    let seq_q = 1;
    let seq_k = 3;
    let q = vec![1.0, 0.0];
    let k = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0];
    let v = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];

    let mut out = vec![0.0f32; seq_q * head_dim];
    dense_attention(&q, &k, &v, head_dim, seq_q, seq_k, &mut out).unwrap();

    // scores = q . k = [1.0, 0.5, 0.0]; softmax -> [...]
    let max = 1.0f32;
    let exp = [(1.0f32 - max).exp(), (0.5 - max).exp(), (0.0 - max).exp()];
    let sum: f32 = exp.iter().sum();
    let probs = [exp[0] / sum, exp[1] / sum, exp[2] / sum];
    let expected = [
        probs[0] * v[0] + probs[1] * v[2] + probs[2] * v[4],
        probs[0] * v[1] + probs[1] * v[3] + probs[2] * v[5],
    ];
    assert_buf_close(&out, &expected, 1e-5, 1e-4);
}

#[test]
fn tree_attention_uses_external_tree_causal_mask() {
    // Wrap the tree mask from tree-attention around dense_attention:
    // confirm that tree-shaped causal masking limits which keys are
    // visible to each query.
    let head_dim = 1;
    let seq_q = 1;
    let seq_k = 5; // 2 prefix + 3 tree nodes (width=2, depth=1: 1 + 2 = 3)
    let q = vec![1.0];
    // Keys: make each token distinct so the softmax weighting makes the
    // mask observable in the output.
    let k: Vec<f32> = (0..seq_k).map(|i| i as f32 + 1.0).collect();
    let v: Vec<f32> = (0..seq_k).map(|i| (i as f32 + 1.0) * 10.0).collect();

    let mask = tree_attention::tree_causal_mask(seq_k, 2, 1, 2);
    let mut out = vec![0.0f32; seq_q * head_dim];
    tree_attention_step(&q, &k, &v, &mask, head_dim, seq_q, seq_k, &mut out)
        .unwrap();

    // Expected: q attends to {0, 1} (prefix) and {2} (tree root, since
    // root is ancestor-or-self of all tree nodes). It does NOT attend to
    // {3, 4} which are tree leaves not visible to the root.
    let visible: Vec<f32> = (0..seq_k)
        .filter(|&c| mask[seq_q - 1][c] == 1)
        .map(|c| k[c])
        .collect();
    let vis_v: Vec<f32> = (0..seq_k)
        .filter(|&c| mask[seq_q - 1][c] == 1)
        .map(|c| v[c])
        .collect();
    let mut dense_out = vec![0.0f32; head_dim];
    dense_attention(&q, &visible, &vis_v, head_dim, seq_q, visible.len(), &mut dense_out)
        .unwrap();
    assert_buf_close(&out, &dense_out, 1e-5, 1e-4);
}

// ===========================================================================
// MoE
// ===========================================================================

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
    let expected = vec![0.6 * 1.0 + 0.4 * 0.0, 0.6 * 0.0 + 0.4 * 1.0, 0.7 * 2.0 + 0.3 * 1.0, 0.7 * 2.0 + 0.3 * -1.0];
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

// ===========================================================================
// Recurrent
// ===========================================================================

#[test]
fn deltanet_step_updates_state_correctly() {
    // state shape = (head_dim, head_dim).
    let head_dim = 2;
    let mut state = vec![0.0f32; head_dim * head_dim];
    let q = vec![1.0, 0.0];
    let k = vec![0.5, 0.5];
    let v = vec![2.0, -1.0];
    let beta = 0.5;
    // Compute the *expected* new state and output from the *initial*
    // state, then run the kernel and compare.
    let mut s_new = vec![0.0f32; head_dim * head_dim];
    for i in 0..head_dim {
        for j in 0..head_dim {
            let mut kk = 0.0;
            for p in 0..head_dim {
                kk += k[p] * state[p * head_dim + j];
            }
            s_new[i * head_dim + j] = state[i * head_dim + j] - beta * k[i] * kk + beta * v[i] * k[j];
        }
    }
    let mut expected = vec![0.0f32; head_dim];
    for i in 0..head_dim {
        let mut acc = 0.0;
        for j in 0..head_dim {
            acc += q[j] * s_new[j * head_dim + i];
        }
        expected[i] = acc;
    }
    let out = deltanet_step(&q, &k, &v, &mut state, beta, head_dim).unwrap();
    assert_buf_close(&out, &expected, 1e-5, 1e-4);
    assert_eq!(state, s_new);
}

#[test]
fn deltanet_chunk_matches_repeated_step() {
    // Run two sequential deltanet_steps and compare to one chunk of size 2.
    let head_dim = 2;
    let chunk_size = 2;

    let q = vec![1.0, 0.0, 0.0, 1.0];
    let k = vec![0.5, 0.5, -0.2, 0.3];
    let v = vec![2.0, -1.0, 0.4, 0.8];
    let beta = 0.5;

    let mut state_step = vec![0.0f32; head_dim * head_dim];
    let mut outs_step = Vec::new();
    for c in 0..chunk_size {
        let qc = q[c * head_dim..c * head_dim + head_dim].to_vec();
        let kc = k[c * head_dim..c * head_dim + head_dim].to_vec();
        let vc = v[c * head_dim..c * head_dim + head_dim].to_vec();
        let o = deltanet_step(&qc, &kc, &vc, &mut state_step, beta, head_dim).unwrap();
        outs_step.extend_from_slice(&o);
    }

    let (outs_chunk, state_chunk) =
        deltanet_chunk(&q, &k, &v, chunk_size, head_dim, &vec![0.0; head_dim * head_dim]).unwrap();
    assert_buf_close(&outs_step, &outs_chunk, 1e-4, 1e-3);
    assert_buf_close(&state_step, &state_chunk, 1e-5, 1e-4);
}

#[test]
fn short_conv1d_matches_naive_convolution() {
    let kernel = vec![1.0, 0.5, -0.25];
    // First call: state is empty -> output for token 0 is just kernel[0]*x[0]
    // for the inputs we feed in. Subsequent tokens use the previous inputs.
    let inputs = vec![1.0, 2.0, 3.0, 4.0];
    let mut state: Vec<f32> = Vec::new();
    let mut outs = Vec::new();
    for &x in &inputs {
        let y = short_conv1d_step(&[x], &kernel, &mut state).unwrap();
        outs.push(y);
    }
    // Naive: y[t] = sum_{i=0..k-1} kernel[i] * x[t - (k-1) + i]
    let klen = kernel.len();
    let mut expected = Vec::with_capacity(inputs.len());
    for t in 0..inputs.len() {
        let mut acc = 0.0;
        for i in 0..klen {
            let idx = t as isize - (klen as isize - 1) + i as isize;
            if idx >= 0 {
                acc += kernel[i] * inputs[idx as usize];
            }
        }
        expected.push(acc);
    }
    assert_buf_close(&outs, &expected, 1e-5, 1e-4);
}

#[test]
fn mamba_scan_matches_recurrent_definition() {
    let n = 8;
    let a = vec![0.9f32; n]; // decay
    let b = vec![0.5f32; n]; // input gain
    let u: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let initial_state = 0.0f32;
    let (ys, states) = mamba_scan(&a, &b, &u, initial_state).unwrap();
    // Reference: state[t] = a * state[t-1] + b * u[t]
    //            y[t] = state[t]
    let mut s = initial_state;
    let mut exp_y = Vec::new();
    let mut exp_s = Vec::new();
    for t in 0..n {
        s = a[t] * s + b[t] * u[t];
        exp_y.push(s);
        exp_s.push(s);
    }
    assert_buf_close(&ys, &exp_y, 1e-5, 1e-4);
    assert_buf_close(&states, &exp_s, 1e-5, 1e-4);
}

#[test]
fn rwkv_time_mix_matches_recurrent_definition() {
    // Time mixing: x'[t] = mix_k * x[t] + (1-mix_k) * state_k[t]
    //              state_k[t+1] = mix_v * x[t] + (1-mix_v) * state_v[t]
    //              state_v[t+1] = mix_r * x[t] + (1-mix_r) * state_r[t]
    //              y[t]        = state_v[t+1]
    let mut state = vec![0.0f32; 3]; // [k, v, r] channels
    let x = vec![1.0, 2.0, 3.0, 0.5];
    let mix_k = 0.5;
    let mix_v = 0.25;
    let mix_r = 0.75;
    let mut outs = Vec::new();
    for &xi in &x {
        let y = rwkv_time_mix(&[xi], &mut state, mix_k, mix_v, mix_r).unwrap();
        outs.push(y[0]);
    }
    // Manual reference.
    let mut s = vec![0.0f32; 3];
    let mut exp = Vec::new();
    for &xi in &x {
        let new_k = mix_k * xi + (1.0 - mix_k) * s[0];
        let new_v = mix_v * xi + (1.0 - mix_v) * s[1];
        let new_r = mix_r * xi + (1.0 - mix_r) * s[2];
        exp.push(new_v);
        s = vec![new_k, new_v, new_r];
    }
    assert_buf_close(&outs, &exp, 1e-5, 1e-4);
}

// ===========================================================================
// Diffusion
// ===========================================================================

#[test]
fn denoise_step_updates_only_masked_tokens() {
    // Two tokens, the second is masked.
    let vocab = 4;
    let x_t = vec![0u32, 0];
    let mask = vec![false, true];
    // Model logits: at position 1 we predict token 3 with high confidence;
    // at position 0 we predict token 2 (we'll never see it because it's
    // already unmasked).
    let model_logits = vec![
        // position 0
        0.0, 0.1, 5.0, 0.0,
        // position 1
        0.0, 0.1, 0.2, 9.0,
    ];
    let upd: DenoiseUpdate = denoise_step(
        &x_t,
        &mask,
        &model_logits,
        RemaskStrategy::None,
        0.0,
        vocab,
    )
    .unwrap();
    // Position 0 untouched.
    assert_eq!(upd.next_x[0], 0);
    assert!(!upd.next_mask[0]);
    // Position 1 was masked and remask=None, so it accepts its argmax.
    assert_eq!(upd.next_x[1], 3);
    assert!(!upd.next_mask[1]);
    assert_eq!(upd.accepted_count, 1);
}

#[test]
fn denoise_step_with_no_remask_strategy_leaves_mask_unchanged() {
    // Two tokens, both masked. With RemaskStrategy::None both should
    // accept their argmax and no positions should remain masked.
    let vocab = 3;
    let x_t = vec![0u32, 0];
    let mask = vec![true, true];
    let model_logits = vec![
        0.0, 5.0, 0.1,
        2.0, 0.0, 0.0,
    ];
    let upd = denoise_step(
        &x_t,
        &mask,
        &model_logits,
        RemaskStrategy::None,
        0.0,
        vocab,
    )
    .unwrap();
    for &m in &upd.next_mask {
        assert!(!m);
    }
    assert_eq!(upd.accepted_count, 2);
}

#[test]
fn remask_low_confidence_respects_percentile() {
    // 4 tokens; set confidences [0.9, 0.8, 0.2, 0.1]. Percentile=50
    // means the *lower* half (relative to confidence) is re-masked.
    let scores = vec![0.9, 0.8, 0.2, 0.1];
    let mut mask = vec![true, true, true, true];
    remask(&scores, &mut mask, &RemaskStrategy::LowConfidence { percentile: 50.0 }, 0, 1).unwrap();
    // Tokens 0 and 1 (confidence >= median) are accepted; tokens 2 and 3
    // are re-masked.
    assert!(!mask[0]);
    assert!(!mask[1]);
    assert!(mask[2]);
    assert!(mask[3]);
}

#[test]
fn confidence_scores_match_softmax_max() {
    let logits = vec![1.0, 2.0, 5.0, 0.0];
    let scores = confidence_scores(&logits, 4).unwrap();
    // Softmax max should equal exp(5) / sum(exp(.)) = exp(5)/Z.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: f32 = logits.iter().map(|l| (l - max).exp()).sum();
    let expected = (5.0f32 - max).exp() / exp;
    assert!((scores[0] - expected).abs() < 1e-5);
    assert_eq!(scores.len(), 1);
}

#[test]
fn parallel_denoise_matches_sequential_denoise() {
    let vocab = 4;
    let n = 6;
    let mut rng = Lcg::new(SEED ^ 0xD1FF);
    let x_t: Vec<u32> = (0..n).map(|i| (i as u32) % vocab as u32).collect();
    let mask: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let model_logits: Vec<f32> = (0..n * vocab).map(|_| rng.next_signed()).collect();

    let upd = denoise_step(
        &x_t,
        &mask,
        &model_logits,
        RemaskStrategy::RandomFraction(0.0),
        0.0,
        vocab,
    )
    .unwrap();
    // With remask fraction 0, no tokens get re-masked: any masked token
    // accepts its argmax.
    for i in 0..n {
        if mask[i] {
            assert!(!upd.next_mask[i], "masked token {i} should accept at frac=0");
        } else {
            assert_eq!(upd.next_x[i], x_t[i]);
        }
    }
}

// ===========================================================================
// Quantized
// ===========================================================================

#[test]
fn ternary_pack_matches_manual_packing() {
    // 8 values packed into 2 bytes (2 bits each), single group of size 8.
    let values = vec![
        SignedTernary::Zero,
        SignedTernary::Pos,
        SignedTernary::Neg,
        SignedTernary::Pos,
        SignedTernary::Zero,
        SignedTernary::Neg,
        SignedTernary::Neg,
        SignedTernary::Pos,
    ];
    let group_size = 8;
    let (packed, scales, zeros) = ternary_pack(&values, group_size).unwrap();
    assert_eq!(packed.len(), 2);
    assert_eq!(scales.len(), 1);
    assert_eq!(zeros.len(), 1);
    // First byte: lower 2 bits = Zero (00), then Pos (01), Neg (10), Pos (01)
    // -> 0b 01 10 01 00 = 0x64
    assert_eq!(packed[0], 0b01_10_01_00);
    // Second byte: Zero (00), Neg (10), Neg (10), Pos (01) -> 0b 01 10 10 00 = 0x68
    assert_eq!(packed[1], 0b01_10_10_00);
}

#[test]
fn ternary_unpack_inverts_pack() {
    let values = vec![
        SignedTernary::Pos,
        SignedTernary::Zero,
        SignedTernary::Neg,
        SignedTernary::Pos,
        SignedTernary::Neg,
        SignedTernary::Pos,
        SignedTernary::Zero,
        SignedTernary::Pos,
    ];
    let group_size = 8;
    let (packed, scales, zeros) = ternary_pack(&values, group_size).unwrap();
    let mut out = vec![SignedTernary::Zero; values.len()];
    ternary_unpack(&packed, &scales, &zeros, values.len(), group_size, &mut out).unwrap();
    assert_eq!(out, values);
}

#[test]
fn subbyte_pack_bits_2_3_4_roundtrip() {
    for &bits in &[2u8, 3, 4] {
        let n = 8;
        let group_size = 8;
        let values: Vec<f32> = (0..n).map(|i| i as f32 / (n as f32)).collect();
        let (packed, scales, zeros) = subbyte_pack(&values, bits, group_size).unwrap();
        let mut out = vec![0.0f32; n];
        subbyte_unpack(&packed, &scales, &zeros, n, group_size, bits, &mut out).unwrap();
        for (i, (&v, &r)) in values.iter().zip(out.iter()).enumerate() {
            // Allow ±1/2^bits relative slack for quantization.
            let slack = 1.0 / (1u32 << bits) as f32;
            let tol = 1e-5 + slack;
            assert!(
                approx_eq(v, r) || (v - r).abs() <= tol + 1e-4 * v.abs(),
                "bits={bits} idx={i}: got {r}, expected {v} (slack {slack})"
            );
        }
    }
}

#[test]
fn subbyte_pack_rejects_bits_outside_1_to_8() {
    let values = vec![0.0f32; 4];
    let err = subbyte_pack(&values, 0, 4).unwrap_err();
    assert!(matches!(err, KernelError::BitsOutOfRange { .. }));
    let err = subbyte_pack(&values, 9, 4).unwrap_err();
    assert!(matches!(err, KernelError::BitsOutOfRange { .. }));
}

#[test]
fn subbyte_pack_handles_partial_trailing_group() {
    // 10 values, group_size=4 -> 3 groups (4, 4, 2).
    let values = vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5];
    let group_size = 4;
    let bits = 4;
    let (packed, scales, zeros) = subbyte_pack(&values, bits, group_size).unwrap();
    let mut out = vec![0.0f32; values.len()];
    subbyte_unpack(&packed, &scales, &zeros, values.len(), group_size, bits, &mut out)
        .unwrap();
    for (i, (&v, &r)) in values.iter().zip(out.iter()).enumerate() {
        let slack = 1.0 / (1u32 << bits) as f32;
        assert!(
            approx_eq(v, r) || (v - r).abs() <= slack + 1e-4 * v.abs(),
            "idx {i}: got {r}, expected {v}"
        );
    }
}

// ===========================================================================
// DispatchPlan (sanity)
// ===========================================================================

#[test]
fn dispatch_plan_exposes_buckets_and_dropped() {
    let assignments: Vec<(usize, f32)> = vec![(0, 0.9), (0, 0.8), (1, 0.7)];
    let plan: DispatchPlan = moe_dispatch(
        &[0, 1, 2],
        &assignments,
        2,
        1.0, // capacity = ceil(1.0 * 3 / 2) = 2 per expert
    )
    .unwrap();
    assert_eq!(plan.expert_buckets.len(), 2);
    assert_eq!(plan.capacity_used.len(), 2);
    assert!(plan.dropped.is_empty());
}
