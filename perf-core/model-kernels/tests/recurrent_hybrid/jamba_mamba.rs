//! Jamba-style hybrid: Mamba selective scan block, then a chunked
//! equivalence assertion. The "attention" half of Jamba is not exercised
//! here — the recurrent block is the novel surface added in this commit;
//! the attention block is covered by the existing dense-attention tests.

use super::*;

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
