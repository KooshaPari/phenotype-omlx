//! Qwen3-Coder-Next + Bonsai acceptance integration tests for
//! `model-kernels`.
//!
//! These tests tie together:
//!
//! - the Bonsai fused ternary matmul (see
//!   `quantized::ternary_matmul::ternary_matmul`);
//! - the Qwen-style DeltaNet chunked linear-recurrent update
//!   (see `recurrent::deltanet`);
//! - the sparse MoE pipeline (see `moe::router`,
//!   `moe::dispatch`, `moe::shared`, `moe::reduce`).
//!
//! The Bonsai row exercises the `Exact ternary block layout and
//! round-trip oracle` row of the model acceptance matrix in
//! `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md`.
//!
//! The Qwen agentic mini-trace exercises the `Long-context decode,
//! tool-use traces, GQA or DeltaNet state, sparse MoE` row of the
//! same matrix. We focus here on the DeltaNet state + sparse MoE
//! pieces; long-context decode and tool-use traces are covered
//! elsewhere in the workspace (see `tests/contracts.rs`).
//!
//! Tolerances follow the crate contract: `abs = 1e-5`, `rel = 1e-4`.
//!
//! The suite is split across per-topic sub-modules:
//!
//! - [`bonsai`] — exact ternary block layout + round-trip oracle.
//! - [`qwen_deltanet`] — Qwen-style DeltaNet chunked linear-recurrent update.
//! - [`qwen_moe`] — sparse MoE pipeline (router / dispatch / shared / reduce).
//! - [`qwen_moe_v2`] — sparse-MoE per-stage composition with the tiled
//!   GEMM, tiled weighted reduce, and dispatch-aware writeback stages.
//! - [`qwen_mini_trace`] — end-to-end agentic DeltaNet + MoE composition.

use model_kernels::common::{approx_eq, Lcg};
use model_kernels::moe::{
    moe_dispatch, router_topk, shared_expert, weighted_reduce, DispatchPlan,
};
use model_kernels::quantized::{
    ternary_matmul, ternary_pack, ternary_unpack, SignedTernary,
};
use model_kernels::recurrent::{deltanet_chunk, deltanet_step};

const SEED: u64 = 0xCAFE_BABE_DEAD_BEEF;

/// Build a deterministic vector of `f32` of length `n`.
fn deterministic_vec(n: usize, salt: u64) -> Vec<f32> {
    let mut rng = Lcg::new(SEED ^ salt);
    (0..n).map(|_| rng.next_signed()).collect()
}

/// Element-wise close comparison with explicit tolerances.
fn assert_buf_close(a: &[f32], b: &[f32], abs: f32, rel: f32) {
    assert_eq!(a.len(), b.len(), "buffer length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let ok = diff <= abs || diff <= rel * x.abs().max(y.abs());
        assert!(ok, "buffers differ at {i}: got {x}, expected {y} (abs={abs}, rel={rel})");
    }
}

/// Run a 4-head, head_dim=4 DeltaNet trace for `chunk_size` steps.
/// Each head runs independently with its own initial state and its
/// own slice of (q, k, v). Returns stacked outputs `[chunk, head_dim]`
/// per head, plus the per-head final states.
fn run_qwen_deltanet_trace(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    initial_states: &[Vec<f32>],
    chunk_size: usize,
    num_heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    // The existing kernel operates per head. We stack the per-head
    // outputs into a flat [chunk_size, num_heads, head_dim] buffer.
    let mut outs = vec![0.0f32; chunk_size * num_heads * head_dim];
    let mut final_states = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let qh: Vec<f32> = (0..chunk_size)
            .flat_map(|c| q[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim].iter().copied())
            .collect();
        let kh: Vec<f32> = (0..chunk_size)
            .flat_map(|c| k[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim].iter().copied())
            .collect();
        let vh: Vec<f32> = (0..chunk_size)
            .flat_map(|c| v[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim].iter().copied())
            .collect();
        let (oh, sh) = deltanet_chunk(&qh, &kh, &vh, chunk_size, head_dim, &initial_states[h]).unwrap();
        for c in 0..chunk_size {
            outs[c * num_heads * head_dim + h * head_dim..c * num_heads * head_dim + h * head_dim + head_dim]
                .copy_from_slice(&oh[c * head_dim..c * head_dim + head_dim]);
        }
        final_states.push(sh);
    }
    (outs, final_states)
}

mod bonsai;
mod qwen_deltanet;
mod qwen_mini_trace;
mod qwen_moe;
mod qwen_moe_v2;
