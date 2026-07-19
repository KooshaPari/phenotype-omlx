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
//!
//! The suite is split across per-topic sub-modules:
//!
//! - [`jamba_mamba`] — Jamba-style hybrid: Mamba selective scan block.
//! - [`rwkv7`] — RWKV-7 time-mix trace and state-resume contract.
//! - [`short_conv`] — `short_conv1d_step` end-to-end + state-resume.

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

mod jamba_mamba;
mod rwkv7;
mod short_conv;
