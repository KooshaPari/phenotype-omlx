//! Shared-expert dense matmul (always-on path inside MoE blocks).
//!
//! # Perf invariants (do not regress)
//!
//! This helper is used by `regress-baseline/tests/dispatch_buckets.rs` to
//! obtain a scalar matmul reference for shape-bucketed perf envelopes. It
//! also backs small forward passes in `model-kernels/tests/*` and the
//! GLM/Qwen reference flows. Two invariants keep it cheap:
//!
//! 1. The divisor preamble is capped at `min(x.len(), w.len())` so it
//!    does **not** regress back to an O(total) scan. Without the cap a
//!    single `512×2048×2048` call would iterate a million candidates on
//!    the divisor loop and the regression test would hang for tens of
//!    minutes.
//! 2. The inner matmul iterates `t → kk → j` so the innermost loop walks
//!    contiguous `&w[kk*n..]` / `&mut out[t*n..]` slices; LLVM auto-vectors
//!    the inner dot-product into wide FMA ops.
//!
//! The regression test that pins both invariants is
//! `regress-baseline/tests/dispatch_buckets.rs` (must finish well under
//! 60 s on a single Apple-silicon thread) and
//! `model-kernels/tests/shared_expert_perf.rs` (must finish under 5 s on
//! the same machine for a 512×512×4096 input).
//!
//! **Do not** change the public signature of [`shared_expert`], the inner
//! tile size used by the regression test, or the shape buckets.

use crate::error::{KernelError, Result};

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
    // Infer k as a divisor of `w.len()` that is consistent across rows.
    // The kernel does not know `k` and `n` directly; we instead walk
    // every row against `w`, requiring `w.len() % k == 0` and
    // `out.len() * k == x.len() * w.len() / n`. For simplicity, we
    // require the caller to pass exactly the right number of tokens
    // and infer `k`/`n` from `w.len()` by also accepting `out.len()`.
    // Concretely: k must divide total, and `n` must divide w.len().
    if w.is_empty() {
        return Err(KernelError::ZeroDimension { what: "w", got: 0 });
    }
    // Try every divisor of `x.len()` as candidate k, requiring
    // `w.len()` to also be divisible by `k` so the row-major weight
    // tensor `[k, n]` is well-formed.
    //
    // Perf note: cap `cand` at `min(total, w.len())`. For `cand >
    // w.len()`, `w.len() % cand == w.len()` (non-zero), so the modulo
    // check is wasted work; iterating past that point was the O(total)
    // regression that hung `dispatch_buckets.rs`. See the module-level
    // "Perf invariants" doc-comment.
    let cand_max = total.min(w.len());
    let mut k = None;
    for cand in 1..=cand_max {
        if total % cand != 0 || w.len() % cand != 0 {
            continue;
        }
        let n = w.len() / cand;
        let num_tokens = total / cand;
        if out.len() == num_tokens * n {
            k = Some(cand);
            break;
        }
    }
    let k = k.ok_or_else(|| KernelError::BadBufferLength {
        what: "x/w/out shapes inconsistent",
        expected: total,
        got: w.len(),
    })?;
    let num_tokens = total / k;
    let n = w.len() / k;
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
        for j in 0..n {
            out_row[j] = 0.0;
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
}
