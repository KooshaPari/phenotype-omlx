//! Shared-expert dense matmul (always-on path inside MoE blocks).
//!
//! # Perf invariants (do not regress)
//!
//! This helper is used by `regress-baseline/tests/dispatch_buckets.rs` to
//! obtain a scalar matmul reference for shape-bucketed perf envelopes. It
//! also backs small forward passes in `model-kernels/tests/*` and the
//! GLM/Qwen reference flows. Two invariants keep it cheap:
//!
//! 1. Shape inference is **O(1)** via the closed-form
//!    `k² = |x|·|w| / |out|` (must be a perfect square that divides both
//!    buffers). Do **not** restore a linear scan over `1..=min(|x|,|w|)` —
//!    that was an O(total) regression that hung `dispatch_buckets.rs` for
//!    tens of minutes on large shapes.
//! 2. The inner matmul iterates `t → kk → j` so the innermost loop walks
//!    contiguous `&w[kk*n..]` / `&mut out[t*n..]` slices; LLVM auto-vectors
//!    the inner dot-product into wide FMA ops.
//!
//! The regression test that pins both invariants is
//! `regress-baseline/tests/dispatch_buckets.rs` (must finish well under
//! 60 s on a single Apple-silicon thread) and
//! `model-kernels/tests/shared_expert_perf.rs` (must finish under the
//! declared wall-clock ceiling for a 512×512×4096 input).
//!
//! **Do not** change the public signature of [`shared_expert`], the inner
//! tile size used by the regression test, or the shape buckets.

use crate::error::{KernelError, Result};

/// Integer square root for perfect-square `k²` recovery. Returns `None`
/// when `n` is not a perfect square.
fn isqrt_exact(n: u128) -> Option<usize> {
    if n == 0 {
        return Some(0);
    }
    // f64 is exact for integers up to 2^53; our shapes stay well below that.
    let root = (n as f64).sqrt().round() as u128;
    if root.saturating_mul(root) == n {
        usize::try_from(root).ok()
    } else {
        None
    }
}

/// Compute `out[t, :] = x[t, :] @ w` where `w` is a single dense
/// `[k, n]` matrix applied to every token.
///
/// Layouts:
///
/// - `x` is `[num_tokens, k]`.
/// - `w` is `[k, n]`.
/// - `out` is `[num_tokens, n]`.
#[inline]
pub fn shared_expert(x: &[f32], w: &[f32], out: &mut [f32]) -> Result<()> {
    let total = x.len();
    if total == 0 {
        return Ok(());
    }
    if w.is_empty() {
        return Err(KernelError::ZeroDimension { what: "w", got: 0 });
    }
    if out.is_empty() {
        return Err(KernelError::BadBufferLength {
            what: "x/w/out shapes inconsistent",
            expected: total,
            got: w.len(),
        });
    }

    // Closed form: |out| · k² = |x| · |w|  ⇒  k² = |x|·|w| / |out|.
    let product = (total as u128)
        .checked_mul(w.len() as u128)
        .ok_or(KernelError::BadBufferLength {
            what: "x/w/out shapes inconsistent",
            expected: total,
            got: w.len(),
        })?;
    let out_len = out.len() as u128;
    if product % out_len != 0 {
        return Err(KernelError::BadBufferLength {
            what: "x/w/out shapes inconsistent",
            expected: total,
            got: w.len(),
        });
    }
    let k_sq = product / out_len;
    let k = isqrt_exact(k_sq).ok_or(KernelError::BadBufferLength {
        what: "x/w/out shapes inconsistent",
        expected: total,
        got: w.len(),
    })?;
    if k == 0 || total % k != 0 || w.len() % k != 0 {
        return Err(KernelError::BadBufferLength {
            what: "x/w/out shapes inconsistent",
            expected: total,
            got: w.len(),
        });
    }
    let num_tokens = total / k;
    let n = w.len() / k;
    if out.len() != num_tokens * n {
        return Err(KernelError::BadBufferLength {
            what: "x/w/out shapes inconsistent",
            expected: total,
            got: w.len(),
        });
    }
    // Inner matmul. Loop order is `t → kk → j` so the innermost loop
    // walks contiguous `&w[kk*n..]` / `&mut out[t*n..]` slices; LLVM
    // auto-vectors the inner reduction into wide FMA ops on both
    // Apple-silicon (NEON) and x86 (SSE/AVX) targets. `x[t*k + kk]` is
    // hoisted out of the j-loop so each input element is loaded once
    // and broadcast `n` times. The j-loop is manually unrolled by 4 so
    // a debug build (which does not auto-unroll) still pays one FMA per
    // iteration of the tight inner block. `out` is zero-initialised per
    // row (rather than accumulating into a single `acc`) so the public
    // "out = x @ w" overwrite semantics are preserved.
    for t in 0..num_tokens {
        let x_row_base = t * k;
        let out_row_base = t * n;
        let out_row = &mut out[out_row_base..out_row_base + n];
        for j in out_row.iter_mut() {
            *j = 0.0;
        }
        let n4 = n & !3usize; // largest multiple of 4 not exceeding n
        for kk in 0..k {
            let x_tk = x[x_row_base + kk];
            let w_kj_base = kk * n;
            let out_row = &mut out[out_row_base..out_row_base + n];
            let w_row = &w[w_kj_base..w_kj_base + n];
            let mut j = 0usize;
            while j < n4 {
                out_row[j] += x_tk * w_row[j];
                out_row[j + 1] += x_tk * w_row[j + 1];
                out_row[j + 2] += x_tk * w_row[j + 2];
                out_row[j + 3] += x_tk * w_row[j + 3];
                j += 4;
            }
            while j < n {
                out_row[j] += x_tk * w_row[j];
                j += 1;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_is_noop() {
        let x: [f32; 0] = [];
        let w = [0.0f32; 1];
        let mut out: [f32; 0] = [];
        shared_expert(&x, &w, &mut out).unwrap();
    }

    #[test]
    fn rejects_empty_w() {
        let x = [1.0f32; 2];
        let w: [f32; 0] = [];
        let mut out = [0.0f32; 2];
        let err = shared_expert(&x, &w, &mut out).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn rejects_inconsistent_shapes() {
        let x = [1.0f32; 5]; // not divisible by anything useful
        let w = [1.0f32; 3]; // 3 * 1 / 5 isn't integer
        let mut out = [0.0f32; 1];
        let err = shared_expert(&x, &w, &mut out).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn isqrt_exact_rejects_non_squares() {
        assert_eq!(isqrt_exact(0), Some(0));
        assert_eq!(isqrt_exact(1), Some(1));
        assert_eq!(isqrt_exact(4096 * 4096), Some(4096));
        assert_eq!(isqrt_exact(10), None);
    }

    #[test]
    fn closed_form_k_matches_512x512x4096() {
        let m = 512usize;
        let n = 512usize;
        let k = 4096usize;
        let x = vec![1.0f32; m * k];
        let w = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];
        shared_expert(&x, &w, &mut out).unwrap();
        assert!(out.iter().all(|&v| (v - k as f32).abs() < 1e-3));
    }
}
