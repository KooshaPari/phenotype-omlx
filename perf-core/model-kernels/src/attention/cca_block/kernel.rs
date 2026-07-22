//! ZAYA block-parallel compressed-context attention kernel and oracle.
//!
//! See the parent module docs for the compression axiom and the
//! block-parallel attention algorithm implemented here.

use crate::error::{KernelError, Result};

use super::block::CcaBlock;

/// Block-parallel ZAYA-style compressed-context attention.
///
/// For each block `b` the score is
/// `score_b = (q · block_summary_b) * block_summary_scale_b`.
/// The normalised weight is
/// `w_b = exp(score_b - max_score) / sum_j exp(score_j - max_score)`
/// and the output is the softmax-weighted sum of the block summaries
/// weighted by their size:
/// `out = Σ_b w_b * block_size_b * block_summary_b`.
///
/// Properties enforced by the kernel:
///
/// - An empty `blocks` slice produces an all-zero output (no scores →
///   softmax degenerates to zero weight on each block).
/// - The score is computed with a numerically-stable softmax (subtract
///   the row maximum before exponentiating).
/// - Ties in the score produce deterministic, equal weights (the
///   kernel does not introduce noise or perform any hidden
///   reordering).
/// - `block_summary.len()` must equal `head_dim` for every block and
///   `head_dim` must be strictly positive.
///
/// `out` must have length exactly `head_dim`. Returns the matching
/// [`KernelError`] on a violation.
pub fn cca_block_attend(
    q: &[f32],
    blocks: &[CcaBlock],
    head_dim: usize,
    out: &mut [f32],
) -> Result<()> {
    if head_dim == 0 {
        return Err(KernelError::ZeroDimension { what: "head_dim", got: 0 });
    }
    if q.len() != head_dim {
        return Err(KernelError::BadBufferLength {
            what: "q",
            expected: head_dim,
            got: q.len(),
        });
    }
    if out.len() != head_dim {
        return Err(KernelError::BadBufferLength {
            what: "out",
            expected: head_dim,
            got: out.len(),
        });
    }
    for (i, block) in blocks.iter().enumerate() {
        if block.block_summary.len() != head_dim {
            return Err(KernelError::BadBufferLength {
                what: "block.block_summary",
                expected: head_dim,
                got: block.block_summary.len(),
            });
        }
        if block.block_indices.is_empty() {
            return Err(KernelError::EmptySequence {
                what: "block.block_indices",
            });
        }
        // Suppress the unused index in release builds where the loop
        // body only needs the position for error reporting.
        let _ = i;
    }

    // Zero the output up front so the loop below can accumulate into it.
    for d in out.iter_mut() {
        *d = 0.0;
    }
    if blocks.is_empty() {
        return Ok(());
    }

    // 1) per-block score
    let mut scores = Vec::with_capacity(blocks.len());
    let mut max = f32::NEG_INFINITY;
    for block in blocks {
        let mut dot = 0.0f32;
        for (d, &q_d) in q.iter().enumerate() {
            dot += q_d * block.block_summary[d];
        }
        let s = dot * block.block_summary_scale;
        scores.push(s);
        if s > max {
            max = s;
        }
    }

    // 2) softmax over scores (numerically stable)
    let mut weights = Vec::with_capacity(blocks.len());
    let mut sum = 0.0f32;
    for &s in &scores {
        let e = (s - max).exp();
        weights.push(e);
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for w in weights.iter_mut() {
            *w *= inv;
        }
    } else {
        // Degenerate row (all scores == max but produced zero weight, e.g.
        // exp(-inf)). Fall back to zero-weighted output rather than NaN.
        for w in weights.iter_mut() {
            *w = 0.0;
        }
    }

    // 3) accumulate block_size-weighted summaries into out
    for (block, &w) in blocks.iter().zip(weights.iter()) {
        let scale = w * (block.block_size() as f32);
        for (d, o) in out.iter_mut().enumerate() {
            *o += scale * block.block_summary[d];
        }
    }
    Ok(())
}

/// Reference (oracle) used by the focused tests in [`tests`]. Computes
/// the same ZAYA block-parallel attention by hand: softmax over the
/// per-block scores, then weight each summary by its block size.
pub fn cca_block_attend_oracle(q: &[f32], blocks: &[CcaBlock], head_dim: usize) -> Vec<f32> {
    debug_assert_eq!(q.len(), head_dim);
    let mut out = vec![0.0f32; head_dim];
    if blocks.is_empty() {
        return out;
    }
    let mut scores = Vec::with_capacity(blocks.len());
    let mut max = f32::NEG_INFINITY;
    for block in blocks {
        let mut dot = 0.0f32;
        for (d, &q_d) in q.iter().enumerate() {
            dot += q_d * block.block_summary[d];
        }
        let s = dot * block.block_summary_scale;
        scores.push(s);
        if s > max {
            max = s;
        }
    }
    let mut weights = vec![0.0f32; blocks.len()];
    let mut sum = 0.0f32;
    for (i, &s) in scores.iter().enumerate() {
        let e = (s - max).exp();
        weights[i] = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for w in weights.iter_mut() {
            *w *= inv;
        }
    }
    for (block, &w) in blocks.iter().zip(weights.iter()) {
        let scale = w * (block.block_size() as f32);
        for (d, o) in out.iter_mut().enumerate() {
            *o += scale * block.block_summary[d];
        }
    }
    out
}
