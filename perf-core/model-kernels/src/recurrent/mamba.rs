//! Selective state-space scan (Mamba-style scalar recurrence).
//!
//! Scalar recurrence:
//!
//! ```text
//! state[t] = a[t] * state[t-1] + b[t] * u[t]
//! y[t]     = state[t]
//! ```
//!
//! `a`, `b`, `u` are length-`n` per-time-step coefficients.

use crate::error::{KernelError, Result};

/// Run the Mamba scalar recurrence. Returns the per-step outputs
/// `[n]` and the per-step states `[n]` (so the caller can re-seed
/// the next chunk).
pub fn mamba_scan(
    a: &[f32],
    b: &[f32],
    u: &[f32],
    initial_state: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let n = a.len();
    if n == 0 {
        return Err(KernelError::EmptySequence { what: "a" });
    }
    if b.len() != n || u.len() != n {
        return Err(KernelError::BadBufferLength {
            what: "b/u",
            expected: n,
            got: b.len().max(u.len()),
        });
    }
    let mut state = initial_state;
    let mut ys = Vec::with_capacity(n);
    let mut states = Vec::with_capacity(n);
    for t in 0..n {
        state = a[t] * state + b[t] * u[t];
        ys.push(state);
        states.push(state);
    }
    Ok((ys, states))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty() {
        let err = mamba_scan(&[], &[], &[], 0.0).unwrap_err();
        assert!(matches!(err, KernelError::EmptySequence { .. }));
    }

    #[test]
    fn rejects_length_mismatch() {
        let err = mamba_scan(&[0.5, 0.5], &[1.0], &[1.0, 1.0], 0.0).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn matches_recurrence_one_step() {
        // a=0.5, b=2, u=3, s=0 -> state=6.
        let (ys, _) = mamba_scan(&[0.5], &[2.0], &[3.0], 0.0).unwrap();
        assert!((ys[0] - 6.0).abs() < 1e-5);
    }
}
