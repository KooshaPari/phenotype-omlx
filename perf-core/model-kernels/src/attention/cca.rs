//! Compressed-context attention (ZAYA-style).

use crate::error::{KernelError, Result};

/// Compressed-context attention (ZAYA-style).
///
/// `compressed_k` / `compressed_v` are laid out as
/// `[seq_k / compressed_factor, head_dim]`. Each compressed slot
/// implicitly covers `compressed_factor` logical keys, but the kernel
/// only attends to the `compressed_factor`-length compressed sequence.
pub fn cca_attention(
    compressed_k: &[f32],
    compressed_v: &[f32],
    q: &[f32],
    compressed_factor: usize,
    head_dim: usize,
    seq_q: usize,
    seq_k: usize,
    out: &mut [f32],
) -> Result<()> {
    if head_dim == 0 {
        return Err(KernelError::ZeroDimension { what: "head_dim", got: 0 });
    }
    if compressed_factor == 0 {
        return Err(KernelError::ZeroDimension {
            what: "compressed_factor",
            got: 0,
        });
    }
    if seq_k % compressed_factor != 0 {
        return Err(KernelError::DimMismatch {
            what: "seq_k / compressed_factor",
            expected: seq_k / compressed_factor,
            got: 0,
        });
    }
    let compressed_len = seq_k / compressed_factor;
    if compressed_len == 0 {
        return Err(KernelError::EmptySequence { what: "compressed_len" });
    }
    let expected = compressed_len * head_dim;
    if compressed_k.len() != expected {
        return Err(KernelError::BadBufferLength {
            what: "compressed_k",
            expected,
            got: compressed_k.len(),
        });
    }
    if compressed_v.len() != expected {
        return Err(KernelError::BadBufferLength {
            what: "compressed_v",
            expected,
            got: compressed_v.len(),
        });
    }
    let q_len = seq_q * head_dim;
    if q.len() != q_len {
        return Err(KernelError::BadBufferLength {
            what: "q",
            expected: q_len,
            got: q.len(),
        });
    }
    if out.len() != q_len {
        return Err(KernelError::BadBufferLength {
            what: "out",
            expected: q_len,
            got: out.len(),
        });
    }
    for s in 0..seq_q {
        let q_row = &q[s * head_dim..s * head_dim + head_dim];
        let mut scores = vec![0.0f32; compressed_len];
        let mut max = f32::NEG_INFINITY;
        for k in 0..compressed_len {
            let k_row = &compressed_k[k * head_dim..k * head_dim + head_dim];
            let mut dot = 0.0;
            for d in 0..head_dim {
                dot += q_row[d] * k_row[d];
            }
            scores[k] = dot;
            if dot > max {
                max = dot;
            }
        }
        let mut sum = 0.0f32;
        for (_k, score) in scores.iter_mut().enumerate().take(compressed_len) {
            let e = (*score - max).exp();
            *score = e;
            sum += e;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        let out_row = &mut out[s * head_dim..s * head_dim + head_dim];
        for d in out_row.iter_mut() {
            *d = 0.0;
        }
        for k in 0..compressed_len {
            let p = scores[k] * inv;
            let v_row = &compressed_v[k * head_dim..k * head_dim + head_dim];
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
    fn rejects_non_dividing_factor() {
        let err = cca_attention(&[], &[], &[], 3, 2, 1, 5, &mut []).unwrap_err();
        assert!(matches!(err, KernelError::DimMismatch { .. }));
    }

    #[test]
    fn rejects_zero_compressed_factor() {
        let err = cca_attention(&[], &[], &[], 0, 2, 1, 4, &mut []).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }
}
