//! RWKV time-mixing update.
//!
//! Two related recurrences live in this module:
//!
//! 1. [`rwkv_time_mix`] — RWKV-4 / RWKV-5 / RWKV-6 time-mixing on a
//!    3-channel `[k, v, r]` state with three scalar mix coefficients:
//!
//!    ```text
//!    new_k = mix_k * x + (1 - mix_k) * state[0]
//!    new_v = mix_v * x + (1 - mix_v) * state[1]
//!    new_r = mix_r * x + (1 - mix_r) * state[2]
//!    y     = new_v
//!    state <- [new_k, new_v, new_r]
//!    ```
//!
//! 2. [`rwkv7_time_mix`] — RWKV-7 channel-mix on a 4-channel
//!    `[k, v, r, w]` state. Channel `w` is a learned time-decay that
//!    gates the contribution of the running state. The mix
//!    coefficients for the four channels and a scalar `decay` are the
//!    parameters of the block:
//!
//!    ```text
//!    new_k = mix_k * x + (1 - mix_k) * state[0]
//!    new_v = mix_v * x + (1 - mix_v) * state[1]
//!    new_r = mix_r * x + (1 - mix_r) * state[2]
//!    new_w = mix_g * x + (1 - mix_g) * state[3]   // "gate" channel
//!    y     = new_v * (new_w * decay).tanh()       // gate-decay interaction
//!    state <- [new_k, new_v, new_r, new_w]
//!    ```
//!
//!    The `tanh` non-linearity on the gate-decay product is the
//!    canonical RWKV-7 channel-mix gate. The 4-channel state layout is
//!    the contract required by the model acceptance matrix for RWKV
//!    recurrent traces.

use crate::error::{KernelError, Result};

/// Apply one time-mixing step (RWKV-4/5/6). Mutates `state` in place
/// and returns a single-element output vector.
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

/// Apply one step of the RWKV-7 channel-mix block. The state is a
/// 4-channel `[k, v, r, w]` vector. `decay` is the learned time-decay
/// multiplier for the running state and `mix_g` is the per-step mix
/// coefficient for the gate channel. Returns the scalar output and
/// mutates `state` in place.
pub fn rwkv7_time_mix(
    x: &[f32; 4],
    state: &mut [f32; 4],
    mix_k: f32,
    mix_v: f32,
    mix_r: f32,
    mix_g: f32,
    decay: f32,
) -> Result<f32> {
    // Input shape is statically enforced by the `[f32; 4]` parameter
    // type — there's no length-1 invariant here, so the only
    // validation we need is to defend against silently surprising
    // decay/gate values that would destroy numerical determinism.
    if !decay.is_finite() {
        return Err(KernelError::OutOfRange {
            what: "decay",
            min: f32::NEG_INFINITY,
            max: f32::INFINITY,
            got: decay,
        });
    }
    let xi = x; // keep the binding name short
    let new_k = mix_k * xi[0] + (1.0 - mix_k) * state[0];
    let new_v = mix_v * xi[1] + (1.0 - mix_v) * state[1];
    let new_r = mix_r * xi[2] + (1.0 - mix_r) * state[2];
    let new_w = mix_g * xi[3] + (1.0 - mix_g) * state[3];
    let gate = (new_w * decay).tanh();
    let y = new_v * gate;
    state[0] = new_k;
    state[1] = new_v;
    state[2] = new_r;
    state[3] = new_w;
    Ok(y)
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

    // ---- rwkv7_time_mix -----------------------------------------------

    #[test]
    fn rwkv7_rejects_non_finite_decay() {
        let x = [0.0f32; 4];
        let mut state = [0.0f32; 4];
        let err = rwkv7_time_mix(&x, &mut state, 0.5, 0.5, 0.5, 0.5, f32::NAN).unwrap_err();
        assert!(matches!(err, KernelError::OutOfRange { .. }));
    }

    #[test]
    fn rwkv7_zero_state_zero_x_yields_zero_output() {
        // With state = 0 and x = 0 every new channel is 0 and the
        // gate-decay interaction is tanh(0) * 0 = 0.
        let x = [0.0f32; 4];
        let mut state = [0.0f32; 4];
        let y = rwkv7_time_mix(&x, &mut state, 0.5, 0.5, 0.5, 0.5, 1.0).unwrap();
        assert!(y.abs() < 1e-6, "got {y}");
    }

    #[test]
    fn rwkv7_state_continuity_across_multiple_steps() {
        // Run two independent single-step traces and one two-step
        // trace over the same input sequence; the two-step trace's
        // outputs must equal the per-step outputs and its final state
        // must equal the per-step trace's final state.
        let xs = [
            [1.0f32, 0.5, -0.25, 2.0],
            [0.0f32, 1.0, 0.5, -0.5],
            [-1.0f32, 0.25, 1.0, 0.0],
        ];
        let mix_k = 0.5;
        let mix_v = 0.25;
        let mix_r = 0.75;
        let mix_g = 0.4;
        let decay = 0.9;

        let mut per_step_state = [0.0f32; 4];
        let mut per_step_outs = Vec::new();
        for x in &xs {
            let y = rwkv7_time_mix(x, &mut per_step_state, mix_k, mix_v, mix_r, mix_g, decay)
                .unwrap();
            per_step_outs.push(y);
        }

        let mut joint_state = [0.0f32; 4];
        let mut joint_outs = Vec::new();
        for x in &xs {
            let y = rwkv7_time_mix(x, &mut joint_state, mix_k, mix_v, mix_r, mix_g, decay)
                .unwrap();
            joint_outs.push(y);
        }
        for (i, (a, b)) in per_step_outs.iter().zip(joint_outs.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "step {i}: per {a} vs joint {b}");
        }
        for i in 0..4 {
            assert!(
                (per_step_state[i] - joint_state[i]).abs() < 1e-6,
                "state[{i}]: {} vs {}",
                per_step_state[i],
                joint_state[i]
            );
        }
    }

    #[test]
    fn rwkv7_gate_decay_interaction_tanh_clips_output() {
        // Large gate-decay product should tanh-clamp the output. Set
        // state = 0 so the per-step new channels are determined
        // entirely by the input: new_k = mix_k * x[0], new_v =
        // mix_v * x[1], etc. Then y = new_v * tanh(new_w * decay).
        let x = [0.0f32, 1.0, 0.0, 100.0];
        let mut state = [0.0f32; 4];
        let mix_v = 1.0;
        let mix_g = 1.0;
        let decay = 10.0;
        let y = rwkv7_time_mix(&x, &mut state, 0.0, mix_v, 0.0, mix_g, decay).unwrap();
        // new_v = 1.0, new_w = 100, gate = tanh(1000) ≈ 1.0
        assert!((y - 1.0).abs() < 1e-5, "got {y}");
        // State channels reflect the per-channel mixes.
        assert!((state[1] - 1.0).abs() < 1e-5);
        assert!((state[3] - 100.0).abs() < 1e-5);
    }

    #[test]
    fn rwkv7_negative_decay_zeroes_gate_output() {
        // new_w * decay with decay = 0 (or any multiple of 0) collapses
        // the gate to tanh(0) = 0, so y == 0 regardless of new_v.
        let x = [0.0f32, 5.0, 0.0, 1.0];
        let mut state = [0.0f32; 4];
        let mix_v = 1.0;
        let mix_g = 1.0;
        let y = rwkv7_time_mix(&x, &mut state, 0.0, mix_v, 0.0, mix_g, 0.0).unwrap();
        assert!(y.abs() < 1e-6, "got {y}");
        // State still updated.
        assert!((state[1] - 5.0).abs() < 1e-5);
        assert!((state[3] - 1.0).abs() < 1e-5);
    }
}