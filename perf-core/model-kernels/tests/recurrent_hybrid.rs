//! Integration tests for the hybrid recurrent kernel surface.
//!
//! These tests exercise the model-acceptance matrix from
//! `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md` for
//! the rows "Mamba", "Jamba (hybrid Mamba+attention)", and "RWKV". The
//! intent is to compose small traces that combine the available
//! recurrent kernels and verify the composition against hand-coded
//! references. Each test is a black-box contract — it does not depend
//! on internal structure of any individual kernel.
//!
//! Tolerances for kernel-vs-oracle comparisons are `abs = 1e-5`,
//! `rel = 1e-4` per `crate::common` defaults; long RNN traces relax
//! the relative tolerance slightly, documented per-test.

use model_kernels::common::approx_eq_tol;
use model_kernels::recurrent::{
    mamba_selective_scan, mamba_selective_scan_chunk, rwkv7_time_mix, short_conv1d_step,
    MambaSelectiveParams,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const ABS: f32 = 1e-5;
const REL: f32 = 1e-4;

/// Element-wise equality assertion with the documented tolerances.
fn assert_close(a: &[f32], b: &[f32], abs: f32, rel: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: buffer length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if !approx_eq_tol(x, y, abs, rel) {
            panic!("{ctx}: buffers differ at {i}: got {x}, expected {y} (abs={abs}, rel={rel})");
        }
    }
}

// ===========================================================================
// Jamba-style hybrid: Mamba selective scan block, then a chunked
// equivalence assertion. The "attention" half of Jamba is not exercised
// here — the recurrent block is the novel surface added in this commit;
// the attention block is covered by the existing dense-attention tests.
// ===========================================================================

#[test]
fn jamba_mamba_chunked_output_matches_repeated_single_steps() {
    // 8-token Mamba block with a 4-channel state. We feed the same
    // input twice: once as a single 8-step chunked scan, once as two
    // 4-step chunks that resume from each other's final state.
    let state_dim = 4usize;
    let a_log = [0.1f32, -0.2, 0.05, -0.05];
    let dt = [0.5, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3, 0.2];
    let b = [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4];
    let c = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let u = [1.0, -0.5, 0.25, 2.0, -1.0, 0.5, 0.0, 0.75];
    let params = MambaSelectiveParams {
        dt: &dt,
        a_log: &a_log,
        b: &b,
        c: &c,
        d: &d,
    };

    // Reference: single 8-step chunked scan.
    let mut s_full = vec![0.0f32; state_dim];
    let (full_outs, full_state) =
        mamba_selective_scan_chunk(&params, &u, &mut s_full, 8).unwrap();

    // Hybrid: chunk it as 4 + 4 and verify the per-chunk outputs equal
    // the corresponding slice of the full run, with state continuity.
    // Each chunk receives its own slice of the per-step params.
    let chunk0_params = MambaSelectiveParams {
        dt: &dt[..4],
        a_log: &a_log,
        b: &b[..4],
        c: &c[..4],
        d: &d[..4],
    };
    let (out_a, state_a) =
        mamba_selective_scan_chunk(&chunk0_params, &u[..4], &mut vec![0.0f32; state_dim], 4)
            .unwrap();
    let chunk1_params = MambaSelectiveParams {
        dt: &dt[4..],
        a_log: &a_log,
        b: &b[4..],
        c: &c[4..],
        d: &d[4..],
    };
    let mut s_b = state_a;
    let (out_b, state_b) =
        mamba_selective_scan_chunk(&chunk1_params, &u[4..], &mut s_b, 4).unwrap();

    assert_close(&out_a, &full_outs[..4], ABS, REL, "chunk 1 outs");
    assert_close(&out_b, &full_outs[4..], ABS, REL, "chunk 2 outs");
    assert_close(&state_b, &full_state, ABS, REL, "final state continuity");
}

#[test]
fn jamba_state_resume_after_single_step_matches_chunked_run() {
    // State continuity contract: running a single step, then resuming
    // the same trace with the returned state, must produce the same
    // outputs as running the whole multi-step call from scratch.
    let state_dim = 3usize;
    let a_log = [0.0f32, 0.05, -0.05];
    let dt = [0.2, 0.2, 0.2, 0.2, 0.2];
    let b = [0.5, 0.5, 0.5, 0.5, 0.5];
    let c = [1.0, 1.0, 1.0, 1.0, 1.0];
    let d = [0.25, 0.25, 0.25, 0.25, 0.25];
    let u = [0.1, 0.2, 0.3, 0.4, 0.5];
    let params = MambaSelectiveParams {
        dt: &dt,
        a_log: &a_log,
        b: &b,
        c: &c,
        d: &d,
    };

    let mut s_chunked = vec![0.0f32; state_dim];
    let (chunked_outs, _) =
        mamba_selective_scan_chunk(&params, &u, &mut s_chunked, u.len()).unwrap();

    // Single-step first, then resume for the remaining four. Each
    // call uses a slice of the per-step params sized to match its u.
    let single_params = MambaSelectiveParams {
        dt: &dt[..1],
        a_log: &a_log,
        b: &b[..1],
        c: &c[..1],
        d: &d[..1],
    };
    let mut s_resume = vec![0.0f32; state_dim];
    let first = mamba_selective_scan(&single_params, &u[..1], &mut s_resume).unwrap();
    let rest_params = MambaSelectiveParams {
        dt: &dt[1..],
        a_log: &a_log,
        b: &b[1..],
        c: &c[1..],
        d: &d[1..],
    };
    let rest = mamba_selective_scan(&rest_params, &u[1..], &mut s_resume).unwrap();

    let mut resumed_outs = Vec::with_capacity(u.len());
    resumed_outs.extend_from_slice(&first);
    resumed_outs.extend_from_slice(&rest);

    assert_close(
        &resumed_outs,
        &chunked_outs,
        ABS,
        REL,
        "single-step resume == chunked run",
    );
}

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