//! Mixture-of-experts — single-file facade (spec: `moe.rs`).
//!
//! Re-exports the MoE primitives implemented under `moe/`:
//!
//! - [`router_topk`] — stable top-k expert selection with seeded
//!   tie-break.
//! - [`moe_dispatch`] / [`DispatchPlan`] — capacity-factor-aware token
//!   bucketing per expert.
//! - [`grouped_gemm`] — per-bucket scalar GEMM
//!   (`out = a[bucket] @ b[expert[bucket]]`).
//! - [`weighted_reduce`] — top-k expert output blending.
//! - [`shared_expert`] — dense matmul for always-on shared experts.
//!
//! All routers, dispatchers, and reducers are pure functions of their
//! inputs plus any caller-provided seed. No FFI, no global state.

pub use crate::moe::dispatch::{moe_dispatch, DispatchPlan};
pub use crate::moe::gemm::grouped_gemm;
pub use crate::moe::reduce::weighted_reduce;
pub use crate::moe::router::router_topk;
pub use crate::moe::shared::shared_expert;
