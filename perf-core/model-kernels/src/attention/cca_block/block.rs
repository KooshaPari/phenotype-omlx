//! `CcaBlock` layout for ZAYA-style variable-length compressed blocks.
//!
//! See the parent module docs for the compression axiom and the
//! block-parallel attention algorithm.

/// One compressed CCA block.
///
/// A block summarises `block_indices.len()` raw tokens with a single
/// `head_dim`-long vector (`block_summary`). The scalar
/// `block_summary_scale` lets a model learn per-block temperature /
/// magnitude without re-training `block_summary`.
#[derive(Debug, Clone, PartialEq)]
pub struct CcaBlock {
    /// Learned per-block summary vector, length `== head_dim`.
    pub block_summary: Vec<f32>,
    /// Learned per-block scalar scale applied to the score.
    pub block_summary_scale: f32,
    /// Indices of the raw tokens this block covers. Must be non-empty.
    pub block_indices: Vec<usize>,
}

impl CcaBlock {
    /// Number of raw tokens this block covers.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_indices.len()
    }
}
