//! Contract tests for the `model-kernels` crate.
//!
//! These tests are written TDD-style *before* the corresponding kernels
//! are implemented. Every function call below is a black-box contract:
//! the test does not depend on internal structure. Names match those in
//! `docs/sessions/20260718-metal-model-runtime/07_IMPLEMENTATION_PLAN.md`.
//!
//! Conventions:
//!
//! - Tolerances for oracle vs. kernel comparisons are `abs = 1e-5`,
//!   `rel = 1e-4`. Long RNNs use a slightly looser bound documented
//!   per-test.
//! - Random inputs are produced from a fixed seed (`0xCAFEBABE`).
//! - Buffers are sized to the documented layout per kernel.

use model_kernels::attention::{
    cca_attention, dense_attention, gqa_attention, mla_attention, paged_attention,
    tree_attention_step,
};
use model_kernels::common::{approx_eq, Lcg};
use model_kernels::diffusion::{
    confidence_scores, denoise_step, remask, DenoiseUpdate, RemaskStrategy,
};
use model_kernels::error::KernelError;
use model_kernels::moe::{
    grouped_gemm, moe_dispatch, router_topk, shared_expert, weighted_reduce, DispatchPlan,
};
use model_kernels::quantized::{
    subbyte_pack, subbyte_unpack, ternary_pack, ternary_unpack, SignedTernary,
};
use model_kernels::recurrent::{
    deltanet_chunk, deltanet_step, mamba_scan, rwkv_time_mix, short_conv1d_step,
};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

const SEED: u64 = 0xCAFE_BABE;

/// Build a deterministic vector of `f32` of length `n`.
fn deterministic_vec(n: usize, salt: u64) -> Vec<f32> {
    let mut rng = Lcg::new(SEED ^ salt);
    (0..n).map(|_| rng.next_signed()).collect()
}

/// Compare two buffers element-wise using [`approx_eq`].
fn assert_buf_close(a: &[f32], b: &[f32], abs: f32, rel: f32) {
    assert_eq!(a.len(), b.len(), "buffer length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if !model_kernels::common::approx_eq_tol(x, y, abs, rel) {
            panic!("buffers differ at {i}: got {x}, expected {y} (abs={abs}, rel={rel})");
        }
    }
}

mod attention_dense_sparse;
mod attention_gqa_mla;
mod diffusion;
mod dispatch_plan;
mod moe;
mod quantized;
mod recurrent;
