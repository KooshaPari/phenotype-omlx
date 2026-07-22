//! RWKV-7 trace: 16 sequential [4] inputs compared against a hand-coded
//! recurrence reference. This is the recurrent-traces row of the model
//! acceptance matrix.

use super::*;

// ===========================================================================
// RWKV-7 trace: 16 sequential [4] inputs compared against a hand-coded
// recurrence reference. This is the recurrent-traces row of the model
// acceptance matrix.
// ===========================================================================

#[test]
fn rwkv7_16_step_trace_matches_hand_coded_reference() {
    // 16-step RWKV-7 trace over a fixed sequence of `[4]` inputs.
    let xs: [[f32; 4]; 16] = [
        [0.10, 0.20, -0.05, 0.50],
        [0.30, -0.10, 0.15, 0.40],
        [-0.20, 0.25, 0.05, 0.30],
        [0.05, 0.00, -0.10, 0.20],
        [0.50, 0.50, 0.50, 0.50],
        [-0.50, -0.25, 0.10, -0.10],
        [0.10, 0.10, 0.10, 0.10],
        [0.00, 0.00, 0.00, 0.00],
        [0.20, -0.30, 0.40, -0.50],
        [0.15, 0.25, -0.35, 0.45],
        [0.05, -0.15, 0.25, -0.35],
        [-0.05, 0.15, -0.25, 0.35],
        [0.10, 0.10, -0.10, -0.10],
        [0.30, -0.30, 0.30, -0.30],
        [0.40, 0.40, 0.40, 0.40],
        [-0.10, 0.20, -0.30, 0.40],
    ];
    let mix_k = 0.42;
    let mix_v = 0.27;
    let mix_r = 0.61;
    let mix_g = 0.33;
    let decay = 0.88;

    // Kernel run.
    let mut state = [0.0f32; 4];
    let mut kernel_outs = Vec::with_capacity(xs.len());
    for x in &xs {
        let y = rwkv7_time_mix(x, &mut state, mix_k, mix_v, mix_r, mix_g, decay).unwrap();
        kernel_outs.push(y);
    }

    // Hand-coded reference: replay the documented recurrence.
    let mut ref_state = [0.0f32; 4];
    let mut ref_outs = Vec::with_capacity(xs.len());
    for x in &xs {
        let new_k = mix_k * x[0] + (1.0 - mix_k) * ref_state[0];
        let new_v = mix_v * x[1] + (1.0 - mix_v) * ref_state[1];
        let new_r = mix_r * x[2] + (1.0 - mix_r) * ref_state[2];
        let new_w = mix_g * x[3] + (1.0 - mix_g) * ref_state[3];
        let gate = (new_w * decay).tanh();
        let y = new_v * gate;
        ref_state = [new_k, new_v, new_r, new_w];
        ref_outs.push(y);
    }

    assert_close(&kernel_outs, &ref_outs, ABS, REL, "RWKV-7 16-step trace");

    // State continuity: every channel of the kernel state must match
    // the hand-coded reference state at the end of the trace.
    for i in 0..4 {
        assert!(
            approx_eq_tol(state[i], ref_state[i], ABS, REL),
            "RWKV-7 final state[{i}]: kernel {} vs reference {}",
            state[i],
            ref_state[i]
        );
    }
}

#[test]
fn rwkv7_state_resume_after_first_step_matches_full_trace() {
    // Compose: run step 0 in isolation, capture state, then run the
    // remaining 15 steps starting from that state. The combined
    // outputs must equal running the whole 16-step trace from scratch.
    let xs: [[f32; 4]; 16] = [
        [0.10, 0.20, -0.05, 0.50],
        [0.30, -0.10, 0.15, 0.40],
        [-0.20, 0.25, 0.05, 0.30],
        [0.05, 0.00, -0.10, 0.20],
        [0.50, 0.50, 0.50, 0.50],
        [-0.50, -0.25, 0.10, -0.10],
        [0.10, 0.10, 0.10, 0.10],
        [0.00, 0.00, 0.00, 0.00],
        [0.20, -0.30, 0.40, -0.50],
        [0.15, 0.25, -0.35, 0.45],
        [0.05, -0.15, 0.25, -0.35],
        [-0.05, 0.15, -0.25, 0.35],
        [0.10, 0.10, -0.10, -0.10],
        [0.30, -0.30, 0.30, -0.30],
        [0.40, 0.40, 0.40, 0.40],
        [-0.10, 0.20, -0.30, 0.40],
    ];
    let mix_k = 0.42;
    let mix_v = 0.27;
    let mix_r = 0.61;
    let mix_g = 0.33;
    let decay = 0.88;

    // Full trace.
    let mut full_state = [0.0f32; 4];
    let mut full_outs = Vec::with_capacity(xs.len());
    for x in &xs {
        let y = rwkv7_time_mix(x, &mut full_state, mix_k, mix_v, mix_r, mix_g, decay).unwrap();
        full_outs.push(y);
    }

    // Step 0 alone.
    let mut resumed_state = [0.0f32; 4];
    let first = rwkv7_time_mix(&xs[0], &mut resumed_state, mix_k, mix_v, mix_r, mix_g, decay)
        .unwrap();
    // Continue from step 1.
    let mut resumed_outs = vec![first];
    for x in &xs[1..] {
        let y =
            rwkv7_time_mix(x, &mut resumed_state, mix_k, mix_v, mix_r, mix_g, decay).unwrap();
        resumed_outs.push(y);
    }

    assert_close(
        &resumed_outs,
        &full_outs,
        ABS,
        REL,
        "RWKV-7 single-step resume == full trace",
    );
    for i in 0..4 {
        assert!(
            approx_eq_tol(resumed_state[i], full_state[i], ABS, REL),
            "RWKV-7 final state[{i}]: resumed {} vs full {}",
            resumed_state[i],
            full_state[i]
        );
    }
}
