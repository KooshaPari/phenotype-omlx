//! Weighted reduction across per-token expert outputs.

use crate::error::{KernelError, Result};

/// Reduce `expert_outs[t, e, :]` into `out[t, :]` by summing
/// `weights[t, e] * expert_outs[t, e, :]`.
///
/// Layouts:
///
/// - `expert_outs` is `[num_tokens, experts_per_token, hidden]`.
/// - `weights` is `[num_tokens, experts_per_token]`.
/// - `out` is `[num_tokens, hidden]` and is fully overwritten.
pub fn weighted_reduce(
    expert_outs: &[f32],
    weights: &[f32],
    experts_per_token: usize,
    hidden: usize,
    out: &mut [f32],
) -> Result<()> {
    if hidden == 0 {
        return Err(KernelError::ZeroDimension { what: "hidden", got: 0 });
    }
    if experts_per_token == 0 {
        return Err(KernelError::ZeroDimension {
            what: "experts_per_token",
            got: 0,
        });
    }
    if weights.is_empty() {
        // No tokens to reduce; nothing to do.
        return Ok(());
    }
    let num_tokens = weights.len() / experts_per_token;
    if weights.len() != num_tokens * experts_per_token {
        return Err(KernelError::BadBufferLength {
            what: "weights",
            expected: weights.len(),
            got: num_tokens * experts_per_token,
        });
    }
    let expected_eo = num_tokens * experts_per_token * hidden;
    if expert_outs.len() != expected_eo {
        return Err(KernelError::BadBufferLength {
            what: "expert_outs",
            expected: expected_eo,
            got: expert_outs.len(),
        });
    }
    if out.len() != num_tokens * hidden {
        return Err(KernelError::BadBufferLength {
            what: "out",
            expected: num_tokens * hidden,
            got: out.len(),
        });
    }
    for t in 0..num_tokens {
        let out_row = &mut out[t * hidden..t * hidden + hidden];
        for h in out_row.iter_mut() {
            *h = 0.0;
        }
        for e in 0..experts_per_token {
            let w = weights[t * experts_per_token + e];
            let eo_row = &expert_outs[(t * experts_per_token + e) * hidden
                ..(t * experts_per_token + e) * hidden + hidden];
            for h in 0..hidden {
                out_row[h] += w * eo_row[h];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_tokens_is_noop() {
        let expert_outs: [f32; 0] = [];
        let weights: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        weighted_reduce(&expert_outs, &weights, 2, 3, &mut out).unwrap();
    }

    #[test]
    fn rejects_zero_hidden() {
        let eo = [0.0f32; 2];
        let w = [1.0f32; 2];
        let mut out = [0.0f32; 0];
        let err = weighted_reduce(&eo, &w, 2, 0, &mut out).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }
}
