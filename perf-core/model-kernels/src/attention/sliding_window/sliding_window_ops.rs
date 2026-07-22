//! Core sliding-window attention compute: the inner unchecked kernel that
//! performs dot-product scoring, masking, softmax, and output accumulation.
//!
//! Split from `sliding_window.rs` to isolate the hot loop from validation
//! and the public API surface.

use crate::attention::common::softmax;

use super::sliding_window_mask::sliding_window_range;

/// Causal sliding-window GQA attention — unchecked (caller must validate).
///
/// `q` is `[seq_q, q_heads, head_dim]`, `k` / `v` are
/// `[seq_k, kv_heads, head_dim]`, `out` is `[seq_q, q_heads, head_dim]`;
/// `window_size` is the per-row width.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sliding_window_attention_unchecked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    seq_q: usize,
    seq_k: usize,
    group_size: usize,
    window_size: usize,
    out: &mut [f32],
) {
    for kh in 0..kv_heads {
        for qh_off in 0..group_size {
            let qh = kh * group_size + qh_off;
            for s in 0..seq_q {
                let (lo, hi) = sliding_window_range(seq_q, seq_k, s, window_size);
                let q_row = &q[s * q_heads * head_dim + qh * head_dim
                    ..s * q_heads * head_dim + qh * head_dim + head_dim];
                let mut scores = vec![f32::NEG_INFINITY; seq_k];
                for t in lo..hi {
                    let k_row = &k[t * kv_heads * head_dim + kh * head_dim
                        ..t * kv_heads * head_dim + kh * head_dim + head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_row[d] * k_row[d];
                    }
                    scores[t] = dot;
                }
                softmax(&mut scores);
                let out_row = &mut out[s * q_heads * head_dim + qh * head_dim
                    ..s * q_heads * head_dim + qh * head_dim + head_dim];
                for d in out_row.iter_mut() {
                    *d = 0.0;
                }
                for t in lo..hi {
                    let p = scores[t];
                    if p == 0.0 {
                        continue;
                    }
                    let v_row = &v[t * kv_heads * head_dim + kh * head_dim
                        ..t * kv_heads * head_dim + kh * head_dim + head_dim];
                    for d in 0..head_dim {
                        out_row[d] += p * v_row[d];
                    }
                }
            }
        }
    }
}
