//! Mixture-of-experts: routing, dispatch, grouped GEMM, shared experts,
//! and reduction.
//!
//! Layouts:
//!
//! - Router logits: `[num_experts]` per token.
//! - Token assignments: `Vec<(expert_id, score)>` produced by the router.
//! - `a`: `[num_tokens, k]` activations.
//! - `b`: `[num_experts, k, n]` expert weights.
//! - `expert_outs`: `[num_tokens, experts_per_token, hidden]` outputs to
//!   reduce.
//!
//! All algorithms in this module are deterministic given the input.

pub mod dispatch;
pub mod gemm;
pub mod gemm_tiled;
pub mod reduce;
pub mod reduce_tiled;
pub mod router;
pub mod shared;

pub use dispatch::{moe_dispatch, DispatchPlan};
pub use gemm::grouped_gemm;
pub use gemm_tiled::grouped_gemm_tiled;
pub use reduce::weighted_reduce;
pub use reduce_tiled::weighted_reduce_tiled;
pub use router::router_topk;
pub use shared::shared_expert;
