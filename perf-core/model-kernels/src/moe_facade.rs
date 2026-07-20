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
//! - [`grouped_gemm_tiled`] — tile/blocked scalar variant of
//!   [`grouped_gemm`]. Same `(a, b, buckets, m, k, n, out)` signature;
//!   iterates the inner reduction block-by-block so the optimizer can
//!   unroll the accumulator. Used by the SOTA candidate
//!   `GroupedGemmMoeTiled` in kernel-registry.
//! - [`weighted_reduce`] — top-k expert output blending.
//! - [`weighted_reduce_tiled`] — tile/blocked scalar variant of
//!   [`weighted_reduce`]. Same
//!   `(expert_outs, weights, experts_per_token, hidden, out)` signature;
//!   iterates the hidden dimension in `tile`-sized blocks
//!   (`tile = min(64, hidden)`). Used by the SOTA candidate
//!   `WeightedMoeReduceTiled` in kernel-registry.
//! - [`shared_expert`] — dense matmul for always-on shared experts.
//! - [`stage_expert_outputs`] / [`coalesced_writeback`] /
//!   [`WritebackPlan`] — dispatch-aware DRAM writeback for the MoE
//!   expert activations. The next kernel in the DAG after
//!   [`weighted_reduce_tiled`]. `stage_expert_outputs` packs the
//!   per-token activations into per-expert contiguous blocks; the
//!   matching `coalesced_writeback` populates the residual stream in
//!   token-major order.
//!
//! All routers, dispatchers, and reducers are pure functions of their
//! inputs plus any caller-provided seed. No FFI, no global state.

pub use crate::moe::dispatch::{moe_dispatch, DispatchPlan};
pub use crate::moe::gemm::grouped_gemm;
pub use crate::moe::gemm_tiled::grouped_gemm_tiled;
pub use crate::moe::reduce::weighted_reduce;
pub use crate::moe::reduce_tiled::weighted_reduce_tiled;
pub use crate::moe::router::router_topk;
pub use crate::moe::shared::shared_expert;
pub use crate::moe::writeback::{coalesced_writeback, stage_expert_outputs, WritebackPlan};
