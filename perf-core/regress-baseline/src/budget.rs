//! Shape-bucketed dispatch / energy budgets.
//!
//! These helpers let a regression test assert "this `(M, N, K)` shape is
//! allowed at most N dispatches and E joules per invocation". They are
//! deliberately stubbed: `dispatch_budget` returns `u64::MAX` and
//! `energy_budget_j` returns `f64::INFINITY`, so the test compiles and
//! runs but does not yet enforce any real ceiling. The follow-up commit
//! (forced by `tests/dispatch_buckets.rs`) populates the body with
//! measurements captured by the test on its first run.

use serde::{Deserialize, Serialize};

/// Operand shape bundle used by [`dispatch_budget`] and [`energy_budget_j`].
///
/// Mirrors the canonical matmul layout (`out[m, n] = a[m, k] @ b[k, n]`)
/// so callers can pin a budget to the actual problem size without
/// dragging in the kernel-registry shape signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShapeKey {
    /// Rows of `a` and `out`.
    pub m: u32,
    /// Columns of `b` and `out`.
    pub n: u32,
    /// Inner-reduction axis shared by `a` and `b`.
    pub k: u32,
}

impl ShapeKey {
    /// Construct a shape key for a dense matmul `(M, N, K)`.
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }

    /// Total output cells (`M * N`). Used by the test to derive the
    /// logical dispatch count when a tiling policy is in play.
    pub fn output_cells(&self) -> u64 {
        self.m as u64 * self.n as u64
    }

    /// Number of fused multiply-add operations the matmul performs
    /// (`2 * M * N * K`). The energy-per-op metric the regression test
    /// reports is `energy_j / flops`, expressed in joules per FLOP.
    pub fn flops(&self) -> u64 {
        2 * self.output_cells() * self.k as u64
    }
}

/// Maximum allowed dispatch count for a single timed invocation of the
/// matmul on `shape`.
///
/// TODO(follow-up): populate this with the measured ceilings observed
/// by `tests/dispatch_buckets.rs` on its first run. The current stub
/// returns `u64::MAX` so no assertion trips before that follow-up lands.
pub fn dispatch_budget(shape: &ShapeKey) -> u64 {
    let _ = shape;
    u64::MAX
}

/// Maximum allowed joules-per-FLOP for a single timed invocation of the
/// matmul on `shape`.
///
/// TODO(follow-up): populate this with the measured ceilings observed
/// by `tests/dispatch_buckets.rs` on its first run. The current stub
/// returns `f64::INFINITY` so no assertion trips before that follow-up
/// lands.
pub fn energy_budget_j(shape: &ShapeKey) -> f64 {
    let _ = shape;
    f64::INFINITY
}
