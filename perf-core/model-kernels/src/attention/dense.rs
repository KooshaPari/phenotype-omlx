//! Vanilla dense attention (single head).

use crate::attention::common::{check_seq_buffer, dense_attention_unchecked};
use crate::error::{KernelError, Result};

/// Vanilla dense attention (single head).
///
/// `q` is `[seq_q, head_dim]`, `k` and `v` are `[seq_k, head_dim]`, `out`
/// is `[seq_q, head_dim]`. Causal masking is *not* applied — the caller
/// passes the key length they want attended to.
pub fn dense_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    seq_q: usize,
    seq_k: usize,
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
    check_seq_buffer("q", q, seq_q, head_dim)?;
    check_seq_buffer("k", k, seq_k, head_dim)?;
    check_seq_buffer("v", v, seq_k, head_dim)?;
    check_seq_buffer("out", out, seq_q, head_dim)?;
    dense_attention_unchecked(q, k, v, head_dim, seq_q, seq_k, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_head_dim() {
        let q = [0.0f32; 1];
        let k = [0.0f32; 1];
        let v = [0.0f32; 1];
        let mut out = [0.0f32; 1];
        let err = dense_attention(&q, &k, &v, 0, 1, 1, &mut out).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn rejects_empty_seq() {
        let q: [f32; 0] = [];
        let k = [0.0f32; 1];
        let v = [0.0f32; 1];
        let mut out: [f32; 0] = [];
        let err = dense_attention(&q, &k, &v, 1, 0, 1, &mut out).unwrap_err();
        assert!(matches!(err, KernelError::EmptySequence { .. }));
    }

    #[test]
    fn single_token_single_dim_picks_value() {
        let q = [0.5f32];
        let k = [0.5f32];
        let v = [1.0f32];
        let mut out = [0.0f32];
        dense_attention(&q, &k, &v, 1, 1, 1, &mut out).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-5);
    }
}
