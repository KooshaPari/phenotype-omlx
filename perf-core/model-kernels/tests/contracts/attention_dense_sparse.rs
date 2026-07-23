//! CCA + paged + dense + tree attention contract tests.
//!
//! Split out of the original Attention section (510 lines, over the
//! 500-line cap). Covers ZAYA block-parallel attention (CCA), paged
//! block-gather attention, the dense baseline, and tree-causal mask
//! conformance for speculative-decoding use.

use super::*;

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
    tree_attention_step(&q, &k, &v, &mask, head_dim, seq_q, seq_k, &mut out).unwrap();

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
    dense_attention(
        &q,
        &visible,
        &vis_v,
        head_dim,
        seq_q,
        visible.len(),
        &mut dense_out,
    )
    .unwrap();
    assert_buf_close(&out, &dense_out, 1e-5, 1e-4);
}
