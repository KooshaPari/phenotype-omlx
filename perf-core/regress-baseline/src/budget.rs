//! Shape-bucketed dispatch / energy budgets.
//!
//! Every canonical [`ShapeKey`] the regression suite tracks has its own
//! `dispatch_budget` (maximum allowed number of Metal command-buffer
//! dispatches) and `energy_budget_j` (maximum allowed joules-per-FLOP).
//! The ceilings are derived from the first observed run of the
//! `tests/dispatch_buckets.rs` envelope test on 2026-07-18, then anchored
//! with explicit headroom (1.2× for dispatches, 1.5× for energy so the
//! per-tile wall-time measurement — which doesn't yet account for
//! memory-bandwidth stalls — can be tightened later without re-bucketing).
//!
//! See [`tests/dispatch_buckets/main.rs`](../../tests/dispatch_buckets/main.rs)
//! for the original measurement and the follow-up plumbing plan.
//!
//! ## Lookup algorithm
//!
//! Buckets are stored in `BUCKETS`, ordered by `output_cells` ascending.
//! `dispatch_budget` / `energy_budget_j` pick the **smallest matching
//! bucket** whose `(m, n, k)` shape key is `>=` the requested shape on
//! every dimension (i.e. the first bucket that strictly contains the
//! request). A request that *exceeds* every bucket (no envelope covers
//! it) returns the largest bucket's ceiling — this matches the spec's
//! "no measured envelope is treated as 0 dispatches", and surfaces the
//! need for a follow-up bucket rather than silently failing.
//!
//! A request whose shape is *smaller* than the smallest bucket is clamped
//! to that smallest bucket's ceiling rather than returning `0`, so a
//! regression test that uses a smaller bucket than the canonical eight
//! still has a useful ceiling.

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

/// One row of the canonical ceiling table.
///
/// `dispatch_ceiling` and `energy_per_op_ceiling_j` correspond to a
/// `(M, N, K)` matmul at the size stored in `shape`. Both ceilings were
/// captured on 2026-07-18 from the first observed run of the
/// `dispatch_buckets.rs` envelope test on this machine, with the
/// documented 1.2× / 1.5× headroom applied.
#[derive(Debug, Clone, Copy)]
struct Bucket {
    shape: ShapeKey,
    dispatch_ceiling: u64,
    energy_per_op_ceiling_j: f64,
}

/// Eight canonical buckets, ordered by `output_cells` ascending:
///
/// | Bucket                              |  dispatches | energy_per_op_j |
/// |-------------------------------------|------------:|----------------:|
/// | longctx_64x32_c2048                 |         154 |        2.00e-7  |
/// | tiny_decode_512x2048x2048           |         308 |        1.75e-7  |
/// | small_prompt_1024x4096x4096         |        1229 |        1.70e-7  |
/// | medium_prompt_2048x8192x8192        |        4916 |        1.80e-7  |
/// | square_4k_4096x4096x4096            |        4916 |        1.80e-7  |
/// | bigmoe_expert_2x14336               |        8602 |        2.00e-7  |
/// | square_8k_8192x8192x8192            |       19661 |        1.90e-7  |
/// | long_decode_16384x4096x4096         |       19661 |        1.95e-7  |
///
/// `longctx_64x32_c2048` is a long-context single-token decode on a
/// Qwen-class 7B at 32 k context: `(M=64, N=8192, K=2048)`. It anchors
/// the very-small `M` end of the envelope (output cells ≈ 524 k) so a
/// skinny decode path has its own ceiling rather than collapsing to the
/// 512×2048 prompt-decode bucket.
///
/// `bigmoe_expert_2x14336` is the heavy Mixtral-class MoE expert FFN
/// forward: `(M=2048, N=14336, K=14336)`. It slots between `square_4k`
/// and `square_8k` (output cells ≈ 29 M) and pins the ceiling for the
/// routed-expert GEMM that dominates MoE inference cost.
const BUCKETS: &[Bucket] = &[
    Bucket {
        shape: ShapeKey::new(64, 8192, 2048),
        dispatch_ceiling: 154,
        energy_per_op_ceiling_j: 2.00e-7,
    },
    Bucket {
        shape: ShapeKey::new(512, 2048, 2048),
        dispatch_ceiling: 308,
        energy_per_op_ceiling_j: 1.75e-7,
    },
    Bucket {
        shape: ShapeKey::new(1024, 4096, 4096),
        dispatch_ceiling: 1229,
        energy_per_op_ceiling_j: 1.70e-7,
    },
    Bucket {
        shape: ShapeKey::new(2048, 8192, 8192),
        dispatch_ceiling: 4916,
        energy_per_op_ceiling_j: 1.80e-7,
    },
    Bucket {
        shape: ShapeKey::new(4096, 4096, 4096),
        dispatch_ceiling: 4916,
        energy_per_op_ceiling_j: 1.80e-7,
    },
    Bucket {
        shape: ShapeKey::new(2048, 14336, 14336),
        dispatch_ceiling: 8602,
        energy_per_op_ceiling_j: 2.00e-7,
    },
    Bucket {
        shape: ShapeKey::new(8192, 8192, 8192),
        dispatch_ceiling: 19661,
        energy_per_op_ceiling_j: 1.90e-7,
    },
    Bucket {
        shape: ShapeKey::new(16384, 4096, 4096),
        dispatch_ceiling: 19661,
        energy_per_op_ceiling_j: 1.95e-7,
    },
];

/// Internal helper: find the bucket that covers `shape`. Returns the
/// smallest bucket whose `(M, N, K)` is componentwise `>=` the request,
/// or the largest bucket if the request exceeds every entry.
///
/// Exposed at module-private scope so the helper test in
/// `tests/budget_coverage.rs` can verify the precedence order without
/// reaching back through the public surface.
fn bucket_for(shape: &ShapeKey) -> &'static Bucket {
    let mut last = BUCKETS.last().expect("non-empty BUCKETS table");
    for b in BUCKETS {
        if shape.m <= b.shape.m && shape.n <= b.shape.n && shape.k <= b.shape.k {
            return b;
        }
        last = b;
    }
    last
}

/// Maximum allowed dispatch count for a single timed invocation of the
/// matmul on `shape`.
///
/// Pulls the per-shape ceiling from `BUCKETS` and falls back to the
/// largest bucket's ceiling if `shape` is bigger than every entry, so a
/// regression test that runs at an unmeasured size still fails loudly
/// (rather than silently returning `u64::MAX`) but does not trip the
/// envelope gate without evidence.
pub fn dispatch_budget(shape: &ShapeKey) -> u64 {
    bucket_for(shape).dispatch_ceiling
}

/// Maximum allowed joules-per-FLOP for a single timed invocation of the
/// matmul on `shape`.
///
/// Same lookup rule as [`dispatch_budget`].
pub fn energy_budget_j(shape: &ShapeKey) -> f64 {
    bucket_for(shape).energy_per_op_ceiling_j
}

#[cfg(test)]
mod bucket_tests {
    use super::*;

    #[test]
    fn exact_shape_match_finds_bucket() {
        let k = ShapeKey::new(4096, 4096, 4096);
        assert_eq!(dispatch_budget(&k), 4916);
        assert!((energy_budget_j(&k) - 1.80e-7).abs() < 1e-15);
    }

    #[test]
    fn smaller_request_clamps_to_smallest_bucket() {
        // Smaller than the smallest bucket (longctx_64x32_c2048): clamps
        // to it rather than 0 so a sub-bucket regression test still has a
        // useful ceiling.
        let k = ShapeKey::new(1, 1, 1);
        assert_eq!(dispatch_budget(&k), 154);
    }

    #[test]
    fn larger_request_falls_back_to_largest_bucket() {
        // 32k square: not a canonical bucket, falls back to the
        // long_decode (16384) ceiling with the explicit "no measured
        // envelope covers this" semantics.
        let k = ShapeKey::new(32768, 32768, 32768);
        assert_eq!(dispatch_budget(&k), 19661);
    }

    #[test]
    fn buckets_are_ordered_by_output_cells() {
        let mut prev = 0u64;
        for b in BUCKETS {
            let cells = b.shape.output_cells();
            assert!(
                cells >= prev,
                "BUCKETS must be ordered by output_cells ascending: {} -> {} is a regression",
                prev,
                cells,
            );
            prev = cells;
        }
    }
}
