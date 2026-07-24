//! LFM2-style gated short convolution.
//!
//! On each call the kernel reads `x`, taps into the carried-over
//! `state` of the previous `kernel.len() - 1` inputs, computes the
//! standard 1-D convolution, and shifts the state forward by one
//! input. The output is a single `f32`.

use crate::error::{KernelError, Result};

/// Apply one step of a short convolution. `state` holds the
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
    window[..expected_state_len].copy_from_slice(&state[..expected_state_len]);
    window[expected_state_len] = x[0];
    let mut acc = 0.0;
    for i in 0..kernel.len() {
        acc += kernel[i] * window[i];
    }
    // Shift state left and append the new input.
    state[..expected_state_len].copy_from_slice(&window[1..(expected_state_len + 1)]);
    Ok(acc)
}

/// Apply one step of an LFM2-style *gated* short convolution.
///
/// Runs two parallel `short_conv1d_step` traces over the same scalar
/// input `x[0]`: one with the value kernel (`kernel`) and one with the
/// gate kernel (`gate_kernel`), each carrying its own state buffer.
/// The returned value is the element-wise product `y_conv * y_gate`,
/// mirroring the LFM2 gating formulation. `gate_state` is updated in
/// place exactly like `state`.
pub fn gated_short_conv1d_step(
    x: &[f32],
    kernel: &[f32],
    gate_kernel: &[f32],
    state: &mut Vec<f32>,
    gate_state: &mut Vec<f32>,
) -> Result<f32> {
    let y_conv = short_conv1d_step(x, kernel, state)?;
    let y_gate = short_conv1d_step(x, gate_kernel, gate_state)?;
    Ok(y_conv * y_gate)
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

    // ---------------- Gated short convolution (LFM2) ----------------

    /// Oracle: gated output equals element-wise product of two separate
    /// `short_conv1d_step` calls driven in lockstep (with synchronised
    /// history state) over the same input sequence.
    #[test]
    fn gated_equals_product_of_two_short_conv_steps() {
        // Two independent pairs of state buffers, each pair locked to
        // the same input sequence but replayed independently of the
        // gated path. The reference never shares state with the gated
        // kernel.
        let mut ref_state: Vec<f32> = Vec::new();
        let mut ref_gate: Vec<f32> = Vec::new();

        let kernel = [1.0f32, 0.5, -0.25];
        let gate_kernel = [0.75f32, -0.5];

        let xs = [1.0f32, 2.0, -1.0, 3.0, -0.5, 0.75];

        // The gated path keeps its own copies of the state buffers
        // (mirroring how a user drives the actual API).
        let mut gated_state: Vec<f32> = Vec::new();
        let mut gated_gate_state: Vec<f32> = Vec::new();

        for (i, &x) in xs.iter().enumerate() {
            let y_gated = gated_short_conv1d_step(
                &[x],
                &kernel,
                &gate_kernel,
                &mut gated_state,
                &mut gated_gate_state,
            )
            .unwrap();

            // Reference: drive both ungated paths on the same input
            // sequence up to and including step `i`.
            let y_conv_ref = short_conv1d_step(&[x], &kernel, &mut ref_state).unwrap();
            let y_gate_ref = short_conv1d_step(&[x], &gate_kernel, &mut ref_gate).unwrap();
            let y_ref = y_conv_ref * y_gate_ref;
            assert!(
                (y_gated - y_ref).abs() <= 1e-5,
                "step {i}: gated={y_gated}, ref={y_ref} (y_conv={y_conv_ref}, y_gate={y_gate_ref})",
            );
        }
    }

    #[test]
    fn gated_propagates_inner_kernel_error() {
        let mut state: Vec<f32> = Vec::new();
        let mut gate_state: Vec<f32> = Vec::new();
        let err =
            gated_short_conv1d_step(&[1.0], &[], &[1.0], &mut state, &mut gate_state).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn gated_state_initialised_lazily() {
        let mut state: Vec<f32> = Vec::new();
        let mut gate_state: Vec<f32> = Vec::new();
        let y =
            gated_short_conv1d_step(&[2.0], &[3.0], &[4.0], &mut state, &mut gate_state).unwrap();
        // single-tap: y_conv = 3 * 2 = 6, y_gate = 4 * 2 = 8, gated = 48.
        assert_eq!(y, 48.0);
        assert!(state.is_empty()); // single-tap → no carry state.
        assert!(gate_state.is_empty());
    }
}
