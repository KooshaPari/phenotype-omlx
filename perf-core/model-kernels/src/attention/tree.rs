//! Tree-shaped attention driven by an externally-computed mask.

use crate::error::{KernelError, Result};

/// Tree-shaped attention driven by an externally-computed mask.
///
/// `mask[r][c] == 1` means the query at row `r` may attend to key at
/// column `c`. The caller supplies the mask built by
/// `tree-attention::tree_causal_mask`. `q`, `k`, `v` are single-head
/// `[seq, head_dim]`.
pub fn tree_attention_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[Vec<u8>],
    head_dim: usize,
    seq_q: usize,
    seq_k: usize,
    out: &mut [f32],
) -> Result<()> {
    if head_dim == 0 {
        return Err(KernelError::ZeroDimension { what: "head_dim", got: 0 });
    }
    if seq_q == 0 {
        return Err(KernelError::EmptySequence { what: "seq_q" });
    }
    if seq_k == 0 {
        return Err(KernelError::EmptySequence { what: "seq_k" });
    }
    if mask.len() < seq_q {
        return Err(KernelError::DimMismatch {
            what: "mask rows",
            expected: seq_q,
            got: mask.len(),
        });
    }
    let q_len = seq_q * head_dim;
    let k_len = seq_k * head_dim;
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
    let scale = 1.0 / (head_dim as f32).sqrt();
    for r in 0..seq_q {
        if mask[r].len() != seq_k {
            return Err(KernelError::DimMismatch {
                what: "mask cols",
                expected: seq_k,
                got: mask[r].len(),
            });
        }
        let q_row = &q[r * head_dim..r * head_dim + head_dim];
        let mut scores = vec![f32::NEG_INFINITY; seq_k];
        let mut max = f32::NEG_INFINITY;
        for c in 0..seq_k {
            if mask[r][c] == 0 {
                continue;
            }
            let k_row = &k[c * head_dim..c * head_dim + head_dim];
            let mut dot = 0.0;
            for d in 0..head_dim {
                dot += q_row[d] * k_row[d];
            }
            let sc = dot * scale;
            scores[c] = sc;
            if sc > max {
                max = sc;
            }
        }
        let mut sum = 0.0f32;
        for c in 0..seq_k {
            if scores[c].is_finite() {
                let e = (scores[c] - max).exp();
                scores[c] = e;
                sum += e;
            } else {
                scores[c] = 0.0;
            }
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        let out_row = &mut out[r * head_dim..r * head_dim + head_dim];
        for d in 0..head_dim {
            out_row[d] = 0.0;
        }
        for c in 0..seq_k {
            let p = scores[c] * inv;
            if p == 0.0 {
                continue;
            }
            let v_row = &v[c * head_dim..c * head_dim + head_dim];
            for d in 0..head_dim {
                out_row[d] += p * v_row[d];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mask_row_mismatch() {
        let q = [0.0f32; 1];
        let k = [0.0f32; 2];
        let v = [0.0f32; 2];
        let mask = vec![vec![0u8]; 1]; // seq_q=1, mask row len=1, but seq_k=2
        let mut out = [0.0f32; 1];
        let err = tree_attention_step(&q, &k, &v, &mask, 1, 1, 2, &mut out).unwrap_err();
        assert!(matches!(err, KernelError::DimMismatch { .. }));
    }

    #[test]
    fn empty_visible_keys_returns_zero() {
        // Mask is all zero -> no keys visible -> output zero.
        let q = [1.0f32];
        let k = [0.5f32];
        let v = [7.0f32];
        let mask = vec![vec![0u8; 1]];
        let mut out = [99.0f32];
        tree_attention_step(&q, &k, &v, &mask, 1, 1, 1, &mut out).unwrap();
        assert_eq!(out[0], 0.0);
    }
}
