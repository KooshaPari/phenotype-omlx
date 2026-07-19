//! RWKV time-mixing update.
//!
//! The recurrence on a 3-channel state `[k, v, r]` with a single
//! scalar input `x` is:
//!
//! ```text
//! new_k = mix_k * x + (1 - mix_k) * state[0]
//! new_v = mix_v * x + (1 - mix_v) * state[1]
//! new_r = mix_r * x + (1 - mix_r) * state[2]
//! y     = new_v
//! state <- [new_k, new_v, new_r]
//! ```
//!
//! `x` is a length-1 slice; we return a length-1 output vector so the
//! caller can stack them per token.

use crate::error::{KernelError, Result};

/// Apply one time-mixing step. Mutates `state` in place and returns a
/// single-element output vector.
pub fn rwkv_time_mix(
    x: &[f32],
    state: &mut [f32],
    mix_k: f32,
    mix_v: f32,
    mix_r: f32,
) -> Result<Vec<f32>> {
    if x.len() != 1 {
        return Err(KernelError::BadBufferLength {
            what: "x",
            expected: 1,
            got: x.len(),
        });
    }
    if state.len() != 3 {
        return Err(KernelError::BadBufferLength {
            what: "state",
            expected: 3,
            got: state.len(),
        });
    }
    let xi = x[0];
    let new_k = mix_k * xi + (1.0 - mix_k) * state[0];
    let new_v = mix_v * xi + (1.0 - mix_v) * state[1];
    let new_r = mix_r * xi + (1.0 - mix_r) * state[2];
    state[0] = new_k;
    state[1] = new_v;
    state[2] = new_r;
    Ok(vec![new_v])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_x_length() {
        let mut s = [0.0f32; 3];
        let err = rwkv_time_mix(&[1.0, 2.0], &mut s, 0.5, 0.5, 0.5).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn rejects_wrong_state_length() {
        let mut s = [0.0f32; 2];
        let err = rwkv_time_mix(&[1.0], &mut s, 0.5, 0.5, 0.5).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn zero_state_yields_mix_v_times_x() {
        let mut s = [0.0f32; 3];
        let y = rwkv_time_mix(&[4.0], &mut s, 0.5, 0.25, 0.75).unwrap();
        assert!((y[0] - 1.0).abs() < 1e-5); // 0.25 * 4.0
    }
}
