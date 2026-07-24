//! Multi-latent attention (DeepSeek).

use crate::error::{KernelError, Result};

/// Multi-latent attention (DeepSeek MLA).
///
/// `q_latent` is `[seq_q, d_latent]`, `k_latent` / `v_latent` are
/// `[seq_k, d_latent]`, `q_rope` is `[seq_q, d_rope]`, `k_rope` is
/// `[seq_k, d_rope]`, `out` is `[seq_q, d_latent + d_rope]`.
///
/// The score for query `s` and key `k` is
/// `q_latent[s] . k_latent[k] + q_rope[s] . k_rope[k]`. The output is
/// the latent-only projection of the softmax-weighted values; the rope
/// tail of the output is filled with zeros so callers that need
/// rope-augmented output can fuse downstream.
#[allow(clippy::too_many_arguments)]
pub fn mla_attention(
    q_latent: &[f32],
    k_latent: &[f32],
    v_latent: &[f32],
    q_rope: &[f32],
    k_rope: &[f32],
    d_latent: usize,
    d_rope: usize,
    seq_q: usize,
    seq_k: usize,
    out: &mut [f32],
) -> Result<()> {
    if d_latent == 0 {
        return Err(KernelError::ZeroDimension {
            what: "d_latent",
            got: 0,
        });
    }
    if d_rope == 0 {
        return Err(KernelError::ZeroDimension {
            what: "d_rope",
            got: 0,
        });
    }
    if seq_q == 0 {
        return Err(KernelError::EmptySequence { what: "seq_q" });
    }
    if seq_k == 0 {
        return Err(KernelError::EmptySequence { what: "seq_k" });
    }
    let q_lat = seq_q * d_latent;
    let k_lat = seq_k * d_latent;
    let q_rp = seq_q * d_rope;
    let k_rp = seq_k * d_rope;
    let out_len = seq_q * (d_latent + d_rope);
    if q_latent.len() != q_lat {
        return Err(KernelError::BadBufferLength {
            what: "q_latent",
            expected: q_lat,
            got: q_latent.len(),
        });
    }
    if k_latent.len() != k_lat || v_latent.len() != k_lat {
        return Err(KernelError::BadBufferLength {
            what: "k_latent/v_latent",
            expected: k_lat,
            got: if k_latent.len() != k_lat {
                k_latent.len()
            } else {
                v_latent.len()
            },
        });
    }
    if q_rope.len() != q_rp {
        return Err(KernelError::BadBufferLength {
            what: "q_rope",
            expected: q_rp,
            got: q_rope.len(),
        });
    }
    if k_rope.len() != k_rp {
        return Err(KernelError::BadBufferLength {
            what: "k_rope",
            expected: k_rp,
            got: k_rope.len(),
        });
    }
    if out.len() != out_len {
        return Err(KernelError::BadBufferLength {
            what: "out",
            expected: out_len,
            got: out.len(),
        });
    }
    for s in 0..seq_q {
        let ql = &q_latent[s * d_latent..s * d_latent + d_latent];
        let qr = &q_rope[s * d_rope..s * d_rope + d_rope];
        let mut scores = vec![0.0f32; seq_k];
        let mut max = f32::NEG_INFINITY;
        for t in 0..seq_k {
            let kl = &k_latent[t * d_latent..t * d_latent + d_latent];
            let kr = &k_rope[t * d_rope..t * d_rope + d_rope];
            let mut dot_l = 0.0;
            for d in 0..d_latent {
                dot_l += ql[d] * kl[d];
            }
            let mut dot_r = 0.0;
            for d in 0..d_rope {
                dot_r += qr[d] * kr[d];
            }
            let sc = dot_l + dot_r;
            scores[t] = sc;
            if sc > max {
                max = sc;
            }
        }
        let mut sum = 0.0f32;
        for score in scores.iter_mut().take(seq_k) {
            let e = (*score - max).exp();
            *score = e;
            sum += e;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        let out_row =
            &mut out[s * (d_latent + d_rope)..s * (d_latent + d_rope) + d_latent + d_rope];
        for d in out_row.iter_mut().take(d_latent) {
            *d = 0.0;
        }
        for t in 0..seq_k {
            let p = scores[t] * inv;
            let vl = &v_latent[t * d_latent..t * d_latent + d_latent];
            for d in 0..d_latent {
                out_row[d] += p * vl[d];
            }
        }
        for d in out_row.iter_mut().skip(d_latent).take(d_rope) {
            *d = 0.0;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_d_latent() {
        let err = mla_attention(&[], &[], &[], &[], &[], 0, 4, 1, 1, &mut []).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }
}
