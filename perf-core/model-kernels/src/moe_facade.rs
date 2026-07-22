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

/// Shape-aware grouped GEMM selection based on the measured crossover:
/// small decode/canonical blocks avoid tiled-loop overhead, while larger
/// prefill blocks use the tiled path for cache locality.
#[allow(clippy::too_many_arguments)]
pub fn grouped_gemm_auto(
    a: &[f32],
    b: &[f32],
    buckets: &[Vec<usize>],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) -> crate::error::Result<()> {
    const TILED_WORK_THRESHOLD: usize = 2_000_000;
    let routed_tokens = buckets.iter().map(Vec::len).sum::<usize>().max(m);
    let work = routed_tokens.saturating_mul(k).saturating_mul(n);
    if work >= TILED_WORK_THRESHOLD {
        grouped_gemm_tiled(a, b, buckets, m, k, n, out)
    } else {
        grouped_gemm(a, b, buckets, m, k, n, out)
    }
}
pub use crate::moe::reduce::weighted_reduce;
pub use crate::moe::reduce_tiled::weighted_reduce_tiled;
pub use crate::moe::router::router_topk;
pub use crate::moe::shared::shared_expert;
pub use crate::moe::writeback::{coalesced_writeback, stage_expert_outputs, WritebackPlan};

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(tokens: usize, experts: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<Vec<usize>>) {
        let a = vec![1.0; tokens * k];
        let b = vec![1.0; experts * k * n];
        let buckets = (0..experts)
            .map(|expert| (0..tokens).filter(|token| token % experts == expert).collect())
            .collect();
        (a, b, buckets)
    }

    #[test]
    fn auto_selection_preserves_oracle_for_small_and_large_shapes() {
        for (tokens, k, n) in [(128, 64, 64), (512, 128, 128)] {
            let (a, b, buckets) = inputs(tokens, 8, k, n);
            let mut expected = vec![0.0; tokens * n];
            let mut actual = vec![0.0; tokens * n];
            grouped_gemm(&a, &b, &buckets, 0, k, n, &mut expected).unwrap();
            grouped_gemm_auto(&a, &b, &buckets, 0, k, n, &mut actual).unwrap();
            assert_eq!(actual, expected);
        }
    }
}
