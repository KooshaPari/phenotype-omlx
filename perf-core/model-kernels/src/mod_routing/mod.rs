//! Mixture-of-Depths (MoD) sparse token routing.
//!
//! MoD (Raposo et al., 2024) is a sparse routing scheme that decides, on
//! every transformer layer, **which tokens** are allowed to enter the
//! block's heavy compute path. Unlike MoE — which selects *which expert*
//! processes each token — MoD selects *which tokens* are processed at all;
//! the surviving tokens share a single expert (or a small fixed stack of
//! experts) while the rest are carried around unchanged.
//!
//! This module is the scalar oracle for the family. It is intentionally
//! minimal:
//!
//! - [`mod_route`] reads per-token router weights, applies a sigmoid
//!   score, and picks the top-`k` survivors by score with ties broken by
//!   *lower index* (the determinism contract used by the kernel
//!   registry's selector).
//! - [`mod_apply`] materializes a contiguous `[k, dim]` buffer of the
//!   surviving rows so downstream kernels can run dense compute.
//! - [`mod_scatter_back`] is the inverse: it scatters the processed rows
//!   back into a `[num_tokens, dim]` buffer and fills the skipped
//!   positions with a caller-supplied constant (typically `0.0`).
//!
//! All functions are pure: no RNG, no FFI, no `unsafe`. The reference
//! scoring function is deterministic so test oracles can be reproduced
//! byte-for-byte. The capacity-factor argument lives in `(0, 1]`;
//! `1.0` means "route every token" (identity) and `0 < c < 1` selects
//! `floor(c * num_tokens)` survivors.
//!
//! # Layout
//!
//! - `weights: [num_tokens]` per-token router weights (typically a small
//!   linear projection of the hidden state followed by a sigmoid).
//! - `full_hidden_states: [num_tokens * dim]` row-major.
//! - `ModRoutePlan::selected_tokens`: survivor indices in input order.
//! - `ModRoutePlan::capacity_factor`: the capacity the plan was built
//!   under; downstream kernels may scale residual streams by
//!   `1 / capacity_factor` to preserve the unconditional expectation
//!   of the per-token contribution.

mod route;
mod apply;

pub use route::{ModRoutePlan, ModRouterConfig, mod_route};
pub use apply::{mod_apply, mod_scatter_back};
// Re-exported so the `use super::*;` inside the `mod tests` block can
// resolve `KernelError`, matching the original flat-file layout where
// the parent module had `use crate::error::{KernelError, Result};` at
// scope.
#[cfg(test)]
pub(crate) use crate::error::KernelError;

#[cfg(test)]
mod tests;
