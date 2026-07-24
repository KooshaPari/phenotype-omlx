//! Section "Recurrent" of the original contracts.rs.
//!
//! Split out of the original monolithic `model-kernels/tests/contracts.rs`
//! (1130 lines) so each topic stays under the 350-line target. Test bodies
//! are byte-identical to the source file; only the surrounding module
//! wrapper and `use super::*;` import differ.

use super::*;

#[test]
fn deltanet_step_updates_state_correctly() {
    // state shape = (head_dim, head_dim).
    let head_dim = 2;
    let mut state = vec![0.0f32; head_dim * head_dim];
    let q = vec![1.0, 0.0];
    let k = vec![0.5, 0.5];
    let v = vec![2.0, -1.0];
    let beta = 0.5;
    // Compute the *expected* new state and output from the *initial*
    // state, then run the kernel and compare.
    let mut s_new = vec![0.0f32; head_dim * head_dim];
    for i in 0..head_dim {
        for j in 0..head_dim {
            let mut kk = 0.0;
            for p in 0..head_dim {
                kk += k[p] * state[p * head_dim + j];
            }
            s_new[i * head_dim + j] =
                state[i * head_dim + j] - beta * k[i] * kk + beta * v[i] * k[j];
        }
    }
    let mut expected = vec![0.0f32; head_dim];
    for i in 0..head_dim {
        let mut acc = 0.0;
        for j in 0..head_dim {
            acc += q[j] * s_new[j * head_dim + i];
        }
        expected[i] = acc;
    }
    let out = deltanet_step(&q, &k, &v, &mut state, beta, head_dim).unwrap();
    assert_buf_close(&out, &expected, 1e-5, 1e-4);
    assert_eq!(state, s_new);
}

#[test]
fn deltanet_chunk_matches_repeated_step() {
    // Run two sequential deltanet_steps and compare to one chunk of size 2.
    let head_dim = 2;
    let chunk_size = 2;

    let q = vec![1.0, 0.0, 0.0, 1.0];
    let k = vec![0.5, 0.5, -0.2, 0.3];
    let v = vec![2.0, -1.0, 0.4, 0.8];
    let beta = 0.5;

    let mut state_step = vec![0.0f32; head_dim * head_dim];
    let mut outs_step = Vec::new();
    for c in 0..chunk_size {
        let qc = q[c * head_dim..c * head_dim + head_dim].to_vec();
        let kc = k[c * head_dim..c * head_dim + head_dim].to_vec();
        let vc = v[c * head_dim..c * head_dim + head_dim].to_vec();
        let o = deltanet_step(&qc, &kc, &vc, &mut state_step, beta, head_dim).unwrap();
        outs_step.extend_from_slice(&o);
    }

    let (outs_chunk, state_chunk) = deltanet_chunk(
        &q,
        &k,
        &v,
        chunk_size,
        head_dim,
        &vec![0.0; head_dim * head_dim],
    )
    .unwrap();
    assert_buf_close(&outs_step, &outs_chunk, 1e-4, 1e-3);
    assert_buf_close(&state_step, &state_chunk, 1e-5, 1e-4);
}

#[test]
fn short_conv1d_matches_naive_convolution() {
    let kernel = vec![1.0, 0.5, -0.25];
    // First call: state is empty -> output for token 0 is just kernel[0]*x[0]
    // for the inputs we feed in. Subsequent tokens use the previous inputs.
    let inputs = vec![1.0, 2.0, 3.0, 4.0];
    let mut state: Vec<f32> = Vec::new();
    let mut outs = Vec::new();
    for &x in &inputs {
        let y = short_conv1d_step(&[x], &kernel, &mut state).unwrap();
        outs.push(y);
    }
    // Naive: y[t] = sum_{i=0..k-1} kernel[i] * x[t - (k-1) + i]
    let klen = kernel.len();
    let mut expected = Vec::with_capacity(inputs.len());
    for (t, _) in inputs.iter().enumerate() {
        let mut acc = 0.0;
        for (i, k) in kernel.iter().enumerate().take(klen) {
            let idx = t as isize - (klen as isize - 1) + i as isize;
            if idx >= 0 {
                acc += k * inputs[idx as usize];
            }
        }
        expected.push(acc);
    }
    assert_buf_close(&outs, &expected, 1e-5, 1e-4);
}

#[test]
fn mamba_scan_matches_recurrent_definition() {
    let n = 8;
    let a = vec![0.9f32; n]; // decay
    let b = vec![0.5f32; n]; // input gain
    let u: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let initial_state = 0.0f32;
    let (ys, states) = mamba_scan(&a, &b, &u, initial_state).unwrap();
    // Reference: state[t] = a * state[t-1] + b * u[t]
    //            y[t] = state[t]
    let mut s = initial_state;
    let mut exp_y = Vec::new();
    let mut exp_s = Vec::new();
    for t in 0..n {
        s = a[t] * s + b[t] * u[t];
        exp_y.push(s);
        exp_s.push(s);
    }
    assert_buf_close(&ys, &exp_y, 1e-5, 1e-4);
    assert_buf_close(&states, &exp_s, 1e-5, 1e-4);
}

#[test]
fn rwkv_time_mix_matches_recurrent_definition() {
    // Time mixing: x'[t] = mix_k * x[t] + (1-mix_k) * state_k[t]
    //              state_k[t+1] = mix_v * x[t] + (1-mix_v) * state_v[t]
    //              state_v[t+1] = mix_r * x[t] + (1-mix_r) * state_r[t]
    //              y[t]        = state_v[t+1]
    let mut state = vec![0.0f32; 3]; // [k, v, r] channels
    let x = vec![1.0, 2.0, 3.0, 0.5];
    let mix_k = 0.5;
    let mix_v = 0.25;
    let mix_r = 0.75;
    let mut outs = Vec::new();
    for &xi in &x {
        let y = rwkv_time_mix(&[xi], &mut state, mix_k, mix_v, mix_r).unwrap();
        outs.push(y[0]);
    }
    // Manual reference.
    let mut s = vec![0.0f32; 3];
    let mut exp = Vec::new();
    for &xi in &x {
        let new_k = mix_k * xi + (1.0 - mix_k) * s[0];
        let new_v = mix_v * xi + (1.0 - mix_v) * s[1];
        let new_r = mix_r * xi + (1.0 - mix_r) * s[2];
        exp.push(new_v);
        s = vec![new_k, new_v, new_r];
    }
    assert_buf_close(&outs, &exp, 1e-5, 1e-4);
}
