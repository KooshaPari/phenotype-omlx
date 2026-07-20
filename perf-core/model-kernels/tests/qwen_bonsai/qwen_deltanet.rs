//! Qwen3-Coder-Next DeltaNet chunked linear-recurrent acceptance.

use super::*;

// ===========================================================================
// Qwen3-Coder-Next DeltaNet acceptance
// ===========================================================================

#[test]
fn qwen_deltanet_chunk_matches_repeated_step_4_heads() {
    // 4 heads, head_dim=4, 16-step trace.
    let num_heads = 4;
    let head_dim = 4;
    let chunk_size = 16;
    let beta = 0.5;

    let q = deterministic_vec(chunk_size * num_heads * head_dim, 0xD0_1A);
    let k = deterministic_vec(chunk_size * num_heads * head_dim, 0xD0_1B);
    let v = deterministic_vec(chunk_size * num_heads * head_dim, 0xD0_1C);

    // Per-head initial states: deterministic, head-stratified so the
    // four heads are distinguishable.
    let initial_states: Vec<Vec<f32>> = (0..num_heads)
        .map(|h| {
            let salt = SEED ^ (0xE0_F0u64 + h as u64);
            let mut rng = Lcg::new(salt);
            (0..head_dim * head_dim).map(|_| rng.next_signed() * 0.25).collect()
        })
        .collect();

    // Path A: chunk via deltanet_chunk per head.
    let (chunk_outs, chunk_states) = run_qwen_deltanet_trace(
        &q,
        &k,
        &v,
        &initial_states,
        chunk_size,
        num_heads,
        head_dim,
    );

    // Path B: run deltanet_step sequentially per head and stack.
    let mut step_states: Vec<Vec<f32>> = initial_states.clone();
    let mut step_outs = vec![0.0f32; chunk_size * num_heads * head_dim];
    for h in 0..num_heads {
        for c in 0..chunk_size {
            let qc = &q[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim];
            let kc = &k[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim];
            let vc = &v[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim];
            let o = deltanet_step(qc, kc, vc, &mut step_states[h], beta, head_dim).unwrap();
            step_outs[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim]
                .copy_from_slice(&o);
        }
    }

    // Per-head outputs and final states must match.
    for h in 0..num_heads {
        let mut per_head_chunk = vec![0.0f32; chunk_size * head_dim];
        let mut per_head_step = vec![0.0f32; chunk_size * head_dim];
        for c in 0..chunk_size {
            per_head_chunk[c * head_dim..c * head_dim + head_dim].copy_from_slice(
                &chunk_outs[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim],
            );
            per_head_step[c * head_dim..c * head_dim + head_dim].copy_from_slice(
                &step_outs[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim],
            );
        }
        assert_buf_close(&per_head_chunk, &per_head_step, 1e-5, 1e-4);
        assert_buf_close(&chunk_states[h], &step_states[h], 1e-5, 1e-4);
    }

    // And all entries must be finite.
    assert!(chunk_outs.iter().all(|v| v.is_finite()), "non-finite chunk output");
    for (h, state) in chunk_states.iter().enumerate().take(num_heads) {
        assert!(
            state.iter().all(|v| v.is_finite()),
            "non-finite head {h} state"
        );
    }
}
