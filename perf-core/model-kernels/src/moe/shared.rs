//! Shared-expert dense matmul (always-on path inside MoE blocks).

use crate::error::{KernelError, Result};

/// Compute `out[t, :] = x[t, :] @ w` where `w` is a single dense
/// `[k, n]` matrix applied to every token.
///
/// Layouts:
///
/// - `x` is `[num_tokens, k]`.
/// - `w` is `[k, n]`.
/// - `out` is `[num_tokens, n]`.
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
    let mut k = None;
    for cand in 1..=total {
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
    for t in 0..num_tokens {
        let out_row = &mut out[t * n..t * n + n];
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[t * k + kk] * w[kk * n + j];
            }
            out_row[j] = acc;
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
