//! Grouped-query attention (Qwen3-Coder-Next, Llama-3, etc.).

use crate::attention::common::softmax;
use crate::error::{KernelError, Result};

/// Grouped-query attention.
///
/// `q` is `[seq_q, q_heads, head_dim]`, `k` / `v` are
/// `[seq_k, kv_heads, head_dim]`, `out` is `[seq_q, q_heads, head_dim]`.
/// `group_size = q_heads / kv_heads` is asserted to be exact.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    seq_q: usize,
    seq_k: usize,
    group_size: usize,
    out: &mut [f32],
) -> Result<()> {
    if head_dim == 0 {
        return Err(KernelError::ZeroDimension {
            what: "head_dim",
            got: 0,
        });
    }
    if seq_q == 0 {
        return Err(KernelError::EmptySequence { what: "seq_q" });
    }
    if seq_k == 0 {
        return Err(KernelError::EmptySequence { what: "seq_k" });
    }
    if kv_heads == 0 {
        return Err(KernelError::ZeroDimension {
            what: "kv_heads",
            got: 0,
        });
    }
    if q_heads == 0 {
        return Err(KernelError::ZeroDimension {
            what: "q_heads",
            got: 0,
        });
    }
    if group_size == 0 {
        return Err(KernelError::ZeroDimension {
            what: "group_size",
            got: 0,
        });
    }
    if kv_heads != q_heads / group_size {
        return Err(KernelError::BadGqaGrouping { q_heads, kv_heads });
    }
    let q_len = seq_q * q_heads * head_dim;
    let k_len = seq_k * kv_heads * head_dim;
    if q.len() != q_len {
        return Err(KernelError::BadBufferLength {
            what: "q",
            expected: q_len,
            got: q.len(),
        });
    }
    if k.len() != k_len || v.len() != k_len {
        return Err(KernelError::BadBufferLength {
            what: "k/v",
            expected: k_len,
            got: if k.len() != k_len { k.len() } else { v.len() },
        });
    }
    if out.len() != q_len {
        return Err(KernelError::BadBufferLength {
            what: "out",
            expected: q_len,
            got: out.len(),
        });
    }
    gqa_attention_unchecked(
        q, k, v, q_heads, kv_heads, head_dim, seq_q, seq_k, group_size, out,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gqa_attention_unchecked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    seq_q: usize,
    seq_k: usize,
    group_size: usize,
    out: &mut [f32],
) {
    // No 1/sqrt(d) scaling: parity with `dense_attention` so the
    // `gqa_attention_matches_dense_when_group_size_is_one` contract
    // holds. Callers that want scaled dot-product should rescale Q
    // upstream.
    for kh in 0..kv_heads {
        for qh_off in 0..group_size {
            let qh = kh * group_size + qh_off;
            for s in 0..seq_q {
                let q_row = &q[s * q_heads * head_dim + qh * head_dim
                    ..s * q_heads * head_dim + qh * head_dim + head_dim];
                let mut scores: Vec<f32> = vec![0.0; seq_k];
                for t in 0..seq_k {
                    let k_row = &k[t * kv_heads * head_dim + kh * head_dim
                        ..t * kv_heads * head_dim + kh * head_dim + head_dim];
                    let mut dot = 0.0;
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
                for t in 0..seq_k {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_kv_heads() {
        let err = gqa_attention(&[], &[], &[], 4, 0, 2, 1, 1, 1, &mut []).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn rejects_bad_grouping() {
        // q_heads=4 kv_heads=3 group_size=1 -> kv_heads != q_heads/group_size.
        let q = vec![0.0f32; 4];
        let k = vec![0.0f32; 3];
        let v = vec![0.0f32; 3];
        let mut out = vec![0.0f32; 4];
        let err = gqa_attention(&q, &k, &v, 4, 3, 1, 1, 1, 1, &mut out).unwrap_err();
        assert!(matches!(err, KernelError::BadGqaGrouping { .. }));
    }
}
