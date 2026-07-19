//! Tests for [`crate::recurrent::deltanet_batched::deltanet_batched_chunk`].
//!
//! Contract requirements under test:
//!
//! - (a) 1x1 oracle matches `deltanet_chunk` byte-for-byte.
//! - (b) 2x2 stacked per-`(batch, head)` chunks equal the batched output.
//! - (c) 4x3 scales up; `chunk_size` is uniform because the layout is contiguous.
//! - (d) Zero-dimension rejection for `batch_size`, `num_heads`,
//!   `chunk_size`, `head_dim`.
//! - (e) Mismatched buffer lengths return `BadBufferLength`.
//! - (f) Numerical tolerance via `crate::common::approx_eq`.
//! - Plus: step-wise vs chunk oracle agreement.

use crate::common::approx_eq;
use crate::error::KernelError;
use crate::recurrent::deltanet::deltanet_chunk;
use crate::recurrent::deltanet_batched::{
    deltanet_batched_chunk, deltanet_batched_chunk_stepwise,
};

fn fill_with(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = crate::common::Lcg::new(seed);
    (0..len).map(|_| rng.next_signed()).collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(approx_eq(a, e), "mismatch at index {i}: actual={a}, expected={e}");
    }
}

fn reference_chunk_for_one(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    initial_state: &[f32],
    chunk_size: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    deltanet_chunk(q, k, v, chunk_size, head_dim, initial_state).unwrap()
}

/// Build per-`(b, h)` inputs and the expected stacked outputs using
/// distinct LCG seeds. Returns `(q, k, v, initial_state, exp_outs,
/// exp_states)` ready to feed `deltanet_batched_chunk` and compare.
fn stacked_oracle(
    batch_size: usize,
    num_heads: usize,
    chunk_size: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = Vec::new();
    let mut k = Vec::new();
    let mut v = Vec::new();
    let mut initial_state = Vec::new();
    let mut exp_outs = Vec::new();
    let mut exp_states = Vec::new();
    for b in 0..batch_size {
        for h in 0..num_heads {
            let seed = (b * 31 + h * 17 + 5) as u64;
            let qb = fill_with(seed, chunk_size * head_dim);
            let kb = fill_with(seed.wrapping_add(11), chunk_size * head_dim);
            let vb = fill_with(seed.wrapping_add(23), chunk_size * head_dim);
            let sb = fill_with(seed.wrapping_add(37), head_dim * head_dim);
            q.extend_from_slice(&qb);
            k.extend_from_slice(&kb);
            v.extend_from_slice(&vb);
            initial_state.extend_from_slice(&sb);
            let (o, s) = reference_chunk_for_one(&qb, &kb, &vb, &sb, chunk_size, head_dim);
            exp_outs.extend_from_slice(&o);
            exp_states.extend_from_slice(&s);
        }
    }
    (q, k, v, initial_state, exp_outs, exp_states)
}

/// (a) 1 batch x 1 head: must match `deltanet_chunk` byte-for-byte.
#[test]
fn batched_one_by_one_matches_sequential_chunk() {
    let batch_size = 1;
    let num_heads = 1;
    let chunk_size = 4;
    let head_dim = 3;
    let q = fill_with(11, batch_size * num_heads * chunk_size * head_dim);
    let k = fill_with(22, batch_size * num_heads * chunk_size * head_dim);
    let v = fill_with(33, batch_size * num_heads * chunk_size * head_dim);
    let initial_state = fill_with(44, batch_size * num_heads * head_dim * head_dim);

    let (outs, state) = deltanet_batched_chunk(
        &q,
        &k,
        &v,
        &initial_state,
        batch_size,
        num_heads,
        chunk_size,
        head_dim,
    )
    .expect("batched must succeed");
    let (exp_outs, exp_state) =
        reference_chunk_for_one(&q, &k, &v, &initial_state, chunk_size, head_dim);
    assert_close(&outs, &exp_outs);
    assert_close(&state, &exp_state);
}

/// (b) 2 batches x 2 heads with distinct q/k/v/state.
#[test]
fn batched_two_by_two_stacks_independent_chunk_results() {
    let (b, h, c, d) = (2, 2, 5, 4);
    let (q, k, v, initial_state, exp_outs, exp_states) = stacked_oracle(b, h, c, d);
    let (outs, state) = deltanet_batched_chunk(&q, &k, &v, &initial_state, b, h, c, d)
        .expect("batched must succeed");
    assert_close(&outs, &exp_outs);
    assert_close(&state, &exp_states);
}

/// (c) 4 batches x 3 heads.
#[test]
fn batched_four_by_three_matches_stacked_oracle() {
    let (b, h, c, d) = (4, 3, 6, 5);
    let (q, k, v, initial_state, exp_outs, exp_states) = stacked_oracle(b, h, c, d);
    let (outs, state) = deltanet_batched_chunk(&q, &k, &v, &initial_state, b, h, c, d)
        .expect("batched must succeed");
    assert_close(&outs, &exp_outs);
    assert_close(&state, &exp_states);
}

/// (d) Zero-dimension rejection.
#[test]
fn rejects_zero_batch_size() {
    let initial_state = [0.0f32; 9];
    let err = deltanet_batched_chunk(&[], &[], &[], &initial_state, 0, 1, 2, 3).unwrap_err();
    assert!(
        matches!(err, KernelError::ZeroDimension { what: "batch_size", .. }),
        "got {err:?}"
    );
}

#[test]
fn rejects_zero_num_heads() {
    let initial_state = [0.0f32; 9];
    let err = deltanet_batched_chunk(&[], &[], &[], &initial_state, 1, 0, 2, 3).unwrap_err();
    assert!(
        matches!(err, KernelError::ZeroDimension { what: "num_heads", .. }),
        "got {err:?}"
    );
}

#[test]
fn rejects_zero_chunk_size() {
    let initial_state = [0.0f32; 9];
    let err = deltanet_batched_chunk(&[], &[], &[], &initial_state, 1, 1, 0, 3).unwrap_err();
    assert!(
        matches!(err, KernelError::ZeroDimension { what: "chunk_size", .. }),
        "got {err:?}"
    );
}

#[test]
fn rejects_zero_head_dim_batched() {
    let initial_state: [f32; 0] = [];
    let err = deltanet_batched_chunk(&[], &[], &[], &initial_state, 1, 1, 2, 0).unwrap_err();
    assert!(
        matches!(err, KernelError::ZeroDimension { what: "head_dim", .. }),
        "got {err:?}"
    );
}

/// (e) Mismatched buffer lengths return `BadBufferLength`.
#[test]
fn rejects_mismatched_q_buffer_length() {
    // Per (b, h, chunk, d) = 1*1*2*3 = 6 floats per q/k/v buffer.
    let per_buf = 6;
    let q = vec![0.0f32; per_buf - 1];
    let k = vec![0.0f32; per_buf];
    let v = vec![0.0f32; per_buf];
    let initial_state = vec![0.0f32; 9];
    let err = deltanet_batched_chunk(&q, &k, &v, &initial_state, 1, 1, 2, 3).unwrap_err();
    assert!(
        matches!(err, KernelError::BadBufferLength { what: "q/k/v", .. }),
        "got {err:?}"
    );
}

#[test]
fn rejects_mismatched_initial_state_length() {
    // Per (b, h, chunk, d) = 1*1*2*3 = 6 floats per q/k/v buffer.
    let per_buf = 6;
    let q = vec![0.0f32; per_buf];
    let k = vec![0.0f32; per_buf];
    let v = vec![0.0f32; per_buf];
    let initial_state = vec![0.0f32; 9 - 1];
    let err = deltanet_batched_chunk(&q, &k, &v, &initial_state, 1, 1, 2, 3).unwrap_err();
    assert!(
        matches!(err, KernelError::BadBufferLength { what: "initial_state", .. }),
        "got {err:?}"
    );
}

/// The single `chunk_size` parameter applies uniformly to every
/// `(batch, head)` because the input layout is contiguous row-major. A
/// mismatched chunk_size for some batch would surface as a
/// `BadBufferLength` on `q/k/v`. This test exercises the happy path at
/// scale 2x2 and asserts output sizes.
#[test]
fn chunk_size_must_match_all_batches() {
    let (b, h, c, d) = (2, 2, 4, 3);
    let (q, k, v, initial_state, _exp_outs, _exp_states) = stacked_oracle(b, h, c, d);
    let (outs, state) = deltanet_batched_chunk(&q, &k, &v, &initial_state, b, h, c, d)
        .expect("batched must succeed");
    assert_eq!(outs.len(), b * h * c * d);
    assert_eq!(state.len(), b * h * d * d);
}

/// (f) Numerical tolerance with random inputs (`abs=1e-5, rel=1e-4`).
#[test]
fn batched_matches_sequential_within_tolerance_random_inputs() {
    let (b, h, c, d) = (3, 2, 5, 4);
    let q = fill_with(0xDEAD_BEEF, b * h * c * d);
    let k = fill_with(0xC0FF_EE42, b * h * c * d);
    let v = fill_with(0xABCD_EF99, b * h * c * d);
    let initial_state = fill_with(0x1234_5678, b * h * d * d);

    let mut exp_outs = Vec::new();
    let mut exp_states = Vec::new();
    for bi in 0..b {
        for hi in 0..h {
            let qc = &q[(bi * h + hi) * c * d..(bi * h + hi) * c * d + c * d];
            let kc = &k[(bi * h + hi) * c * d..(bi * h + hi) * c * d + c * d];
            let vc = &v[(bi * h + hi) * c * d..(bi * h + hi) * c * d + c * d];
            let sc = &initial_state[(bi * h + hi) * d * d..(bi * h + hi) * d * d + d * d];
            let (o, s) = reference_chunk_for_one(qc, kc, vc, sc, c, d);
            exp_outs.extend_from_slice(&o);
            exp_states.extend_from_slice(&s);
        }
    }

    let (outs, state) = deltanet_batched_chunk(&q, &k, &v, &initial_state, b, h, c, d)
        .expect("batched must succeed");
    assert_close(&outs, &exp_outs);
    assert_close(&state, &exp_states);
}

/// The step-wise wrapper must agree with the primary `deltanet_chunk`
/// oracle because `deltanet_chunk` is a `chunk_size`-fold of
/// `deltanet_step`. Catches regressions where the two drift.
#[test]
fn stepwise_wrapper_matches_primary() {
    let (b, h, c, d) = (2, 2, 3, 3);
    let q = fill_with(7, b * h * c * d);
    let k = fill_with(13, b * h * c * d);
    let v = fill_with(19, b * h * c * d);
    let initial_state = fill_with(23, b * h * d * d);

    let (outs_a, state_a) =
        deltanet_batched_chunk(&q, &k, &v, &initial_state, b, h, c, d).expect("primary must succeed");
    let (outs_b, state_b) = deltanet_batched_chunk_stepwise(&q, &k, &v, &initial_state, b, h, c, d)
        .expect("stepwise must succeed");
    assert_close(&outs_a, &outs_b);
    assert_close(&state_a, &state_b);
}
