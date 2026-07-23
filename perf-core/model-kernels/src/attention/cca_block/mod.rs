//! Block-parallel compressed-context attention (ZAYA-style).
//!
//! Unlike the uniform-factor `cca_attention` in [`crate::attention::cca`],
//! ZAYA's compressed-context attention handles **variable-length blocks**
//! in one pass: each compressed block covers an arbitrary number of raw
//! tokens (`block_indices.len()`), and the contribution of each block to
//! the final output is weighted by that size — this is the ZAYA
//! "compression axiom": a block that summarises more raw tokens must
//! count proportionally more in the output.
//!
//! The score for block `b` is the dot product of `q` with that block's
//! learned summary vector, scaled by `block_summary_scale`. The
//! normalised per-block weight is then multiplied by `block_size_b`
//! before being accumulated into the output. The caller is responsible
//! for providing a `block_summary` whose length matches `head_dim` and
//! for keeping `block_indices` non-empty (an empty list of raw indices
//! would imply the block represents zero tokens).

mod block;
mod kernel;

pub use block::CcaBlock;
pub use kernel::{cca_block_attend, cca_block_attend_oracle};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::approx_eq;
    use crate::error::KernelError;

    fn assert_buf_close(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(a, e),
                "{label}: mismatch at {i}: got {a}, expected {e}"
            );
        }
    }

    /// Per-block softmax must respect block size: a block that covers
    /// more raw tokens contributes proportionally more to the output
    /// (the ZAYA compression axiom).
    #[test]
    fn per_block_softmax_respects_block_size() {
        let head_dim = 4;
        let q = vec![1.0, 0.5, -0.25, 2.0];
        // Two blocks with identical summaries & scales but different sizes.
        let blocks = vec![
            CcaBlock {
                block_summary: vec![0.5, 1.0, -0.5, 0.25],
                block_summary_scale: 1.0,
                block_indices: (0..4).collect(), // size 4
            },
            CcaBlock {
                block_summary: vec![0.5, 1.0, -0.5, 0.25],
                block_summary_scale: 1.0,
                block_indices: (0..8).collect(), // size 8
            },
        ];
        let mut out = vec![0.0f32; head_dim];
        cca_block_attend(&q, &blocks, head_dim, &mut out).unwrap();
        // Identical summaries + identical scales => equal softmax weights.
        // The output must therefore equal the summary vector scaled by
        // (4 + 8) / 2 = 6, the weighted mean of the two block sizes.
        let expected: Vec<f32> = blocks[0].block_summary.iter().map(|v| v * 6.0).collect();
        assert_buf_close(&out, &expected, "equal-score block_size weighting");
    }

    /// Deterministic tie-break: two blocks with identical scores must
    /// receive identical weights regardless of declaration order.
    #[test]
    fn deterministic_tie_break_for_equal_scores() {
        let head_dim = 2;
        let q = vec![0.0, 0.0]; // dot product is zero for any summary
        let blocks_a = vec![
            CcaBlock {
                block_summary: vec![1.0, 2.0],
                block_summary_scale: 1.0,
                block_indices: vec![0, 1, 2],
            },
            CcaBlock {
                block_summary: vec![3.0, 4.0],
                block_summary_scale: 1.0,
                block_indices: vec![3, 4],
            },
        ];
        let blocks_b = vec![blocks_a[1].clone(), blocks_a[0].clone()];
        let mut out_a = vec![0.0f32; head_dim];
        let mut out_b = vec![0.0f32; head_dim];
        cca_block_attend(&q, &blocks_a, head_dim, &mut out_a).unwrap();
        cca_block_attend(&q, &blocks_b, head_dim, &mut out_b).unwrap();
        // Tied scores ⇒ each block gets weight 1/2. Block sizes are
        // 3 and 2, so total weight × block size = (3 + 2)/2 = 2.5 for
        // every summary dimension ⇒ both orderings must agree.
        assert_buf_close(&out_a, &out_b, "tied scores are order-independent");
        // And the oracle must agree with both.
        let oracle = cca_block_attend_oracle(&q, &blocks_a, head_dim);
        assert_buf_close(&out_a, &oracle, "tied scores vs oracle");
    }

    /// Zero-length block list must return all zeros (no scores ⇒ no
    /// contribution).
    #[test]
    fn zero_length_block_list_returns_zeros() {
        let head_dim = 3;
        let q = vec![1.0, -2.0, 0.5];
        let mut out = vec![99.0f32; head_dim]; // pre-fill to confirm we zero it
        cca_block_attend(&q, &[], head_dim, &mut out).unwrap();
        assert_eq!(out, vec![0.0f32; head_dim]);
    }

    /// A block whose `block_summary` length disagrees with `head_dim`
    /// must be rejected with [`KernelError::BadBufferLength`].
    #[test]
    fn mismatched_head_dim_is_rejected() {
        let head_dim = 4;
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let blocks = vec![CcaBlock {
            block_summary: vec![1.0, 0.0, 0.0], // wrong length
            block_summary_scale: 1.0,
            block_indices: vec![0, 1],
        }];
        let mut out = vec![0.0f32; head_dim];
        let err = cca_block_attend(&q, &blocks, head_dim, &mut out).unwrap_err();
        assert!(
            matches!(err, KernelError::BadBufferLength { .. }),
            "expected BadBufferLength, got {err:?}"
        );
    }

    /// An empty `block_indices` slice is rejected — a block must
    /// cover at least one raw token.
    #[test]
    fn empty_block_indices_is_rejected() {
        let head_dim = 2;
        let q = vec![1.0, 0.0];
        let blocks = vec![CcaBlock {
            block_summary: vec![1.0, 0.0],
            block_summary_scale: 1.0,
            block_indices: Vec::new(),
        }];
        let mut out = vec![0.0f32; head_dim];
        let err = cca_block_attend(&q, &blocks, head_dim, &mut out).unwrap_err();
        assert!(
            matches!(err, KernelError::EmptySequence { .. }),
            "expected EmptySequence, got {err:?}"
        );
    }

    /// `head_dim == 0` is rejected with [`KernelError::ZeroDimension`].
    #[test]
    fn zero_head_dim_is_rejected() {
        let q: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        let err = cca_block_attend(&q, &[], 0, &mut out).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    /// Sanity: kernel output matches the explicit oracle across a small
    /// random-looking trace (deterministic via a fixed seed).
    #[test]
    fn kernel_matches_oracle_on_random_trace() {
        use crate::common::Lcg;
        let head_dim = 8;
        let mut rng = Lcg::new(0xCCAB_B10C);
        let q: Vec<f32> = (0..head_dim).map(|_| rng.next_signed()).collect();
        let sizes = [3usize, 5, 2, 4];
        let mut blocks = Vec::with_capacity(sizes.len());
        for &n in &sizes {
            let summary: Vec<f32> = (0..head_dim).map(|_| rng.next_signed()).collect();
            let scale = rng.next_f32() + 0.5; // positive to avoid sign flips
            let indices: Vec<usize> = (0..n).collect();
            blocks.push(CcaBlock {
                block_summary: summary,
                block_summary_scale: scale,
                block_indices: indices,
            });
        }
        let oracle = cca_block_attend_oracle(&q, &blocks, head_dim);
        let mut kernel_out = vec![0.0f32; head_dim];
        cca_block_attend(&q, &blocks, head_dim, &mut kernel_out).unwrap();
        assert_buf_close(&kernel_out, &oracle, "kernel vs oracle");
    }
}
