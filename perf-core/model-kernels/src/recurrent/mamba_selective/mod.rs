//! Multi-channel selective state-space scan (Mamba-style).
//!
//! Real Mamba uses one scalar state per channel with **selective**
//! (input-dependent) Δ, A, B, C, D parameters. The recurrence is
//!
//! ```text
//! state[c] = exp(dt[t] * exp(a_log[c])) * state[c] + dt[t] * b[t] * u[t]
//! y[t]     = sum_c c[t] * state[c] + d[t] * u[t]
//! ```
//!
//! where
//!
//! - `state` has shape `[state_dim]` (one per channel, not per batch),
//! - `dt` is a length-`n` per-time-step scalar gate,
//! - `a_log` is a length-`state_dim` log-A (Mamba convention:
//!   `A_c = -exp(a_log[c])`), so the per-channel decay is
//!   `exp(dt[t] * exp(a_log[c]))`,
//! - `b`, `c`, `d` are length-`n` per-time-step mix/gain vectors.
//!
//! ## Layout
//!
//! - [`params`]: [`MambaSelectiveParams`] struct.
//! - [`scan`]: [`mamba_selective_scan`] / [`mamba_selective_scan_chunk`]
//!   kernels and their oracle tests.

pub mod params;
pub mod scan;

pub use params::MambaSelectiveParams;
pub use scan::{mamba_selective_scan, mamba_selective_scan_chunk};