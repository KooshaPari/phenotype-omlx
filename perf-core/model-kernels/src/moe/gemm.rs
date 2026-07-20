//! Grouped GEMM: per-bucket dense matmul over `[k, n]` expert weights.
//!
//! Layouts:
//!
//! - `a` is `[num_tokens, k]` activations, row-major.
//! - `b` is `[num_experts, k, n]` expert weights, row-major per expert.
//! - `buckets[e]` is the list of token indices expert `e` owns.
//! - `out` is `[num_tokens, n]` where each row holds `a[tok] @ b[e]`.

use crate::error::{KernelError, Result};

/// Compute `out[tok, :] = a[tok, :] @ b[expert_of(tok), :, :]` for every
/// token in every bucket. Tokens *not* in any bucket are zeroed (and
/// any pre-existing `out` is overwritten for assigned tokens).
///
/// `m` is unused on this scalar path — kept for forward compatibility
/// with future tile sizes. Pass `m = buckets[e].len()` if you want to
/// make this explicit.
#[allow(clippy::too_many_arguments)]
pub fn grouped_gemm(
    a: &[f32],
    b: &[f32],
    buckets: &[Vec<usize>],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    let _ = m; // accepted but unused
    if k == 0 || n == 0 {
        return Err(KernelError::ZeroDimension {
            what: "k or n",
            got: 0,
        });
    }
    // Validate `b` first.
    let expected_b = buckets.len() * k * n;
    if b.len() != expected_b {
        return Err(KernelError::BadBufferLength {
            what: "b",
            expected: expected_b,
            got: b.len(),
        });
    }
    // Validate `a` lazily per row.
    for (e, bucket) in buckets.iter().enumerate() {
        let b_offset = e * k * n;
        for &tok in bucket {
            if tok * k + k > a.len() {
                return Err(KernelError::BadBufferLength {
                    what: "a",
                    expected: (tok + 1) * k,
                    got: a.len(),
                });
            }
            let a_row = &a[tok * k..tok * k + k];
            let out_offset = tok * n;
            if out_offset + n > out.len() {
                return Err(KernelError::BadBufferLength {
                    what: "out",
                    expected: out_offset + n,
                    got: out.len(),
                });
            }
            let out_row = &mut out[out_offset..out_offset + n];
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_row[kk] * b[b_offset + kk * n + j];
                }
                out_row[j] = acc;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_buckets_writes_nothing() {
        let a = [1.0f32; 6];
        let b: [f32; 0] = [];
        let buckets: [Vec<usize>; 0] = [];
        let mut out = [99.0f32; 6];
        grouped_gemm(&a, &b, &buckets, 0, 2, 3, &mut out).unwrap();
        // Out should be untouched.
        assert_eq!(out, [99.0; 6]);
    }

    #[test]
    fn rejects_zero_dim() {
        let a = [0.0f32; 1];
        let b = [0.0f32; 1];
        let buckets = vec![vec![0usize]];
        let mut out = [0.0f32; 1];
        let err = grouped_gemm(&a, &b, &buckets, 1, 0, 1, &mut out).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }
}
