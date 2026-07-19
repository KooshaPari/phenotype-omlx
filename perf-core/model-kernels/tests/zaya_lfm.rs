//! Integration tests for the ZAYA-style block-parallel CCA kernel
//! and the LFM2-style gated short convolution kernel.
//!
//! Both kernels live under `model_kernels::attention::cca_block` and
//! `model_kernels::recurrent::conv` respectively. This file composes
//! end-to-end traces that mirror the model acceptance matrix rows in
//! `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md`
//! for the ZAYA ("CCA and compact nonlinear expert path") and LFM
//! ("Convolution-attention schedule and sparse experts") model
//! families.
//!
//! The integration tests assert against an explicit reference oracle:
//!
//! - For CCA: the reference is `sum_b softmax(q · summary_b *
//!   scale_b) * summary_b * block_size_b`, computed by hand for a
//!   3-block trace over `head_dim == 8`.
//! - For gated short conv: the reference is the element-wise product
//!   of two parallel `short_conv1d_step` traces — one with the value
//!   kernel, one with the gate kernel — over 16 time steps.

use model_kernels::attention::{cca_block_attend, cca_block_attend_oracle, CcaBlock};
use model_kernels::common::approx_eq_tol;
use model_kernels::recurrent::conv::{gated_short_conv1d_step, short_conv1d_step};

/// Compare two buffers element-wise with the crate-wide tolerance
/// contract (`abs = 1e-5`, `rel = 1e-4`).
fn assert_buf_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: buffer length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            approx_eq_tol(a, e, 1e-5, 1e-4),
            "{label}: mismatch at {i}: got {a}, expected {e}"
        );
    }
}

/// Reference oracle for a 3-block ZAYA-style CCA trace. Computed by
/// hand from the formula
/// `out = Σ_b softmax(q · summary_b * scale_b) * summary_b * block_size_b`
/// so the integration test does not depend on the kernel under test.
fn zaya_reference(q: &[f32], blocks: &[CcaBlock], head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; head_dim];
    if blocks.is_empty() {
        return out;
    }
    // Per-block scores.
    let mut scores = Vec::with_capacity(blocks.len());
    let mut max = f32::NEG_INFINITY;
    for block in blocks {
        let mut dot = 0.0f32;
        for (d, &q_d) in q.iter().enumerate().take(head_dim) {
            dot += q_d * block.block_summary[d];
        }
        let s = dot * block.block_summary_scale;
        scores.push(s);
        if s > max {
            max = s;
        }
    }
    // Numerically-stable softmax.
    let mut weights = vec![0.0f32; blocks.len()];
    let mut sum = 0.0f32;
    for (i, &s) in scores.iter().enumerate() {
        let e = (s - max).exp();
        weights[i] = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for w in weights.iter_mut() {
            *w *= inv;
        }
    }
    // Accumulate block_size-weighted summaries.
    for (block, &w) in blocks.iter().zip(weights.iter()) {
        let scale = w * (block.block_indices.len() as f32);
        for (d, out_d) in out.iter_mut().enumerate().take(head_dim) {
            *out_d += scale * block.block_summary[d];
        }
    }
    out
}

#[test]
fn zaya_block_parallel_three_blocks_matches_explicit_reference() {
    // 3 blocks of sizes [4, 2, 6] over head_dim == 8. The reference is
    // computed by hand in `zaya_reference` so this test is independent
    // of `cca_block_attend_oracle`.
    let head_dim = 8usize;
    let q = vec![0.5f32, -0.25, 1.0, -1.5, 0.75, 0.0, -0.5, 2.0];
    let blocks = vec![
        CcaBlock {
            block_summary: vec![1.0f32, 0.5, -0.5, 0.25, 0.0, 0.75, -1.0, 0.5],
            block_summary_scale: 1.0,
            block_indices: (0..4).collect(), // size 4
        },
        CcaBlock {
            block_summary: vec![0.5f32, 1.0, 0.25, -0.75, 1.5, -0.5, 0.0, 0.5],
            block_summary_scale: 0.5, // different scale per block
            block_indices: (0..2).collect(), // size 2
        },
        CcaBlock {
            block_summary: vec![-0.25f32, 0.75, 1.0, -0.5, 0.5, 0.25, -1.0, 1.5],
            block_summary_scale: 1.25, // yet another scale
            block_indices: (0..6).collect(), // size 6
        },
    ];

    let mut actual = vec![0.0f32; head_dim];
    cca_block_attend(&q, &blocks, head_dim, &mut actual).unwrap();
    let reference = zaya_reference(&q, &blocks, head_dim);
    assert_buf_close(
        &actual,
        &reference,
        "ZAYA 3-block trace vs hand-rolled reference",
    );

    // And it must also match the in-crate oracle (two independent
    // implementations of the same spec agree with each other).
    let crate_oracle = cca_block_attend_oracle(&q, &blocks, head_dim);
    assert_buf_close(
        &actual,
        &crate_oracle,
        "ZAYA 3-block trace vs in-crate oracle",
    );
}

#[test]
fn lfm2_gated_short_conv_16_steps_matches_elementwise_product() {
    // 16 inputs through `gated_short_conv1d_step` with a 4-tap value
    // kernel and a 4-tap gate kernel. The gated output must equal
    // `y_conv * y_gate` where `y_conv` and `y_gate` are the outputs
    // of two separate `short_conv1d_step` traces run with identical
    // state management.
    let kernel = [0.5f32, -0.25, 0.125, -0.0625];
    let gate_kernel = [1.0f32, -0.5, 0.25, 0.0];
    let xs: Vec<f32> = (0..16).map(|i| 0.1 * (i as f32) - 0.75).collect();

    // Gated trace.
    let mut state: Vec<f32> = Vec::new();
    let mut gate_state: Vec<f32> = Vec::new();
    let mut gated_outs = Vec::with_capacity(xs.len());
    for &x in &xs {
        let y = gated_short_conv1d_step(&[x], &kernel, &gate_kernel, &mut state, &mut gate_state)
            .unwrap();
        gated_outs.push(y);
    }

    // Reference trace: two parallel ungated short-conv steps.
    let mut ref_state: Vec<f32> = Vec::new();
    let mut ref_gate_state: Vec<f32> = Vec::new();
    let mut ref_conv_outs = Vec::with_capacity(xs.len());
    let mut ref_gate_outs = Vec::with_capacity(xs.len());
    for &x in &xs {
        let y_conv = short_conv1d_step(&[x], &kernel, &mut ref_state).unwrap();
        let y_gate = short_conv1d_step(&[x], &gate_kernel, &mut ref_gate_state).unwrap();
        ref_conv_outs.push(y_conv);
        ref_gate_outs.push(y_gate);
    }

    // Per-step gated output equals the element-wise product of the two
    // reference traces (tolerance is `1e-5` absolute — short conv
    // accumulates only 4 multiplications per step so numerical drift
    // is well within the contract).
    for (i, ((y_gated, &y_conv), &y_gate)) in gated_outs
        .iter()
        .zip(ref_conv_outs.iter())
        .zip(ref_gate_outs.iter())
        .enumerate()
    {
        let expected = y_conv * y_gate;
        assert!(
            (y_gated - expected).abs() <= 1e-5,
            "step {i}: gated ({y_gated}) must equal {y_conv} * {y_gate} = {expected}"
        );
    }

    // The gated state vectors must also match the reference state
    // vectors — this is the documented LFM2 recurrence contract: the
    // gate branch and the value branch carry independent filter
    // histories of the same shape.
    assert_eq!(state, ref_state, "value state must match reference");
    assert_eq!(gate_state, ref_gate_state, "gate state must match reference");
}

#[test]
fn zaya_block_parallel_handles_non_uniform_block_sizes() {
    // Sanity check: a deliberately uneven block-size distribution
    // (sizes [1, 7, 3]) must still produce the same result as the
    // hand-rolled reference, regardless of which block dominates the
    // softmax mass.
    let head_dim = 8usize;
    let q = vec![1.0f32, 0.0, -1.0, 0.5, 0.5, -0.5, 0.0, 0.25];
    let blocks = vec![
        CcaBlock {
            block_summary: vec![0.1f32; head_dim],
            block_summary_scale: 1.0,
            block_indices: vec![0], // size 1
        },
        CcaBlock {
            block_summary: vec![2.0f32, -1.0, 0.5, 0.0, -0.5, 1.0, 1.5, -2.0],
            block_summary_scale: 2.0,
            block_indices: (0..7).collect(), // size 7
        },
        CcaBlock {
            block_summary: vec![-0.5f32, 0.5, -0.25, 1.0, 0.75, -1.0, 0.5, 0.0],
            block_summary_scale: 0.75,
            block_indices: (0..3).collect(), // size 3
        },
    ];

    let mut actual = vec![0.0f32; head_dim];
    cca_block_attend(&q, &blocks, head_dim, &mut actual).unwrap();
    let reference = zaya_reference(&q, &blocks, head_dim);
    assert_buf_close(
        &actual,
        &reference,
        "ZAYA uneven block-size trace vs hand-rolled reference",
    );
}