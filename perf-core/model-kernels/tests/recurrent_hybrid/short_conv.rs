//! short_conv1d_step end-to-end: 32 inputs compared against the explicit
//! convolution reference defined in
//! `model_kernels::recurrent::conv` and exercised in
//! `tests/contracts.rs::short_conv1d_matches_naive_convolution`.

use super::*;

// ===========================================================================
// short_conv1d_step end-to-end: 32 inputs compared against the explicit
// convolution reference defined in
// `model_kernels::recurrent::conv` and exercised in
// `tests/contracts.rs::short_conv1d_matches_naive_convolution`.
// ===========================================================================

#[test]
fn short_conv1d_32_input_trace_matches_naive_convolution() {
    let kernel = [1.0f32, 0.5, -0.25, 0.125];
    let n = 32usize;
    // Build a deterministic input sequence.
    let inputs: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.137).sin() * 2.0).collect();

    // Kernel run.
    let mut state: Vec<f32> = Vec::new();
    let mut outs = Vec::with_capacity(n);
    for &x in &inputs {
        let y = short_conv1d_step(&[x], &kernel, &mut state).unwrap();
        outs.push(y);
    }

    // Naive reference: y[t] = sum_{i=0..k-1} kernel[i] * x[t - (k-1) + i]
    let klen = kernel.len();
    let mut expected = Vec::with_capacity(n);
    for (t, _) in inputs.iter().enumerate() {
        let mut acc = 0.0;
        for (i, &k) in kernel.iter().enumerate().take(klen) {
            let idx = t as isize - (klen as isize - 1) + i as isize;
            if idx >= 0 {
                acc += k * inputs[idx as usize];
            }
        }
        expected.push(acc);
    }
    assert_close(&outs, &expected, ABS, REL, "short_conv1d 32-input trace");
}

#[test]
fn short_conv1d_state_continuity_resume_after_first_input() {
    // Run the first input, capture the kernel's carried state, then
    // resume from that state for the remaining 31 inputs. The combined
    // outputs must equal running the whole 32-input trace from scratch.
    let kernel = [1.0f32, -0.5, 0.25];
    let n = 32usize;
    let inputs: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13).cos()).collect();

    // Full trace.
    let mut full_state: Vec<f32> = Vec::new();
    let mut full_outs = Vec::with_capacity(n);
    for &x in &inputs {
        let y = short_conv1d_step(&[x], &kernel, &mut full_state).unwrap();
        full_outs.push(y);
    }

    // Step 0 then resume.
    let mut resumed_state: Vec<f32> = Vec::new();
    let first = short_conv1d_step(&inputs[..1], &kernel, &mut resumed_state).unwrap();
    let mut resumed_outs = vec![first];
    for &x in &inputs[1..] {
        let y = short_conv1d_step(&[x], &kernel, &mut resumed_state).unwrap();
        resumed_outs.push(y);
    }

    assert_close(
        &resumed_outs,
        &full_outs,
        ABS,
        REL,
        "short_conv1d single-step resume == full trace",
    );
    assert_eq!(
        resumed_state, full_state,
        "short_conv1d final state must match"
    );
}
