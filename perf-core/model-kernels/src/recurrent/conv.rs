//! LFM2-style gated short convolution.
//!
//! On each call the kernel reads `x`, taps into the carried-over
//! `state` of the previous `kernel.len() - 1` inputs, computes the
//! standard 1-D convolution, and shifts the state forward by one
//! input. The output is a single `f32`.

use crate::error::{KernelError, Result};

/// Apply one step of a gated short convolution. `state` holds the
/// most-recent `kernel.len() - 1` scalar inputs; it is updated in
/// place. `x` must have length 1 (one input sample per step).
pub fn short_conv1d_step(x: &[f32], kernel: &[f32], state: &mut Vec<f32>) -> Result<f32> {
    if kernel.is_empty() {
        return Err(KernelError::ZeroDimension {
            what: "kernel",
            got: 0,
        });
    }
    if x.len() != 1 {
        return Err(KernelError::BadBufferLength {
            what: "x",
            expected: 1,
            got: x.len(),
        });
    }
    let expected_state_len = kernel.len() - 1;
    if state.is_empty() {
        state.resize(expected_state_len, 0.0);
    }
    if state.len() != expected_state_len {
        return Err(KernelError::BadBufferLength {
            what: "state",
            expected: expected_state_len,
            got: state.len(),
        });
    }
    // Build the sliding window: [state[0..k-1], x[0]].
    let mut window = vec![0.0f32; kernel.len()];
    for i in 0..expected_state_len {
        window[i] = state[i];
    }
    window[expected_state_len] = x[0];
    let mut acc = 0.0;
    for i in 0..kernel.len() {
        acc += kernel[i] * window[i];
    }
    // Shift state left and append the new input.
    for i in 0..expected_state_len {
        state[i] = window[i + 1];
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_kernel() {
        let mut state: Vec<f32> = Vec::new();
        let err = short_conv1d_step(&[1.0], &[], &mut state).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn rejects_wrong_x_length() {
        let mut state: Vec<f32> = vec![0.0; 2];
        let err = short_conv1d_step(&[1.0, 2.0], &[1.0, 0.5, 0.25], &mut state).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn single_tap_kernel_picks_input() {
        let mut state: Vec<f32> = vec![];
        let y = short_conv1d_step(&[2.0], &[3.0], &mut state).unwrap();
        assert_eq!(y, 6.0);
    }

    #[test]
    fn initializes_empty_state_with_zero_history() {
        let mut state = Vec::new();
        let y = short_conv1d_step(&[1.0], &[1.0, 0.5, -0.25], &mut state).unwrap();
        assert_eq!(y, -0.25);
        assert_eq!(state, vec![0.0, 1.0]);
    }
}
