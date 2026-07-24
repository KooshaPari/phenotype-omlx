//! Tree-attention — explicit causal masks for JetSpec-style draft trees.
//!
//! # Layout
//!
//! A `TreePlan { width, depth }` describes an explicit tree rooted at a single
//! "root" node. Tree nodes are addressed by **in-tree indices** starting at 0:
//!
//! * in-tree index `0` is the root
//! * the parent of in-tree index `i >= 1` is `(i - 1) / width`
//! * in-tree index `i` has children at `width * i + 1 .. width * i + width`
//!
//! When the tree is laid out inside a larger sequence (prefix + draft), the
//! root occupies absolute position `offset` and in-tree index `i` occupies
//! absolute position `offset + i`.
//!
//! # Ancestor-or-self
//!
//! `is_ancestor_or_self(ancestor, descendant)` returns `true` iff `ancestor`
//! is the same node as `descendant` or one of its ancestors in the explicit
//! tree. The algorithm climbs from `descendant` toward the root and stops as
//! soon as the in-tree index drops below `ancestor`, at which point we know
//! we have passed `ancestor` without hitting it. The climb uses the
//! closed-form parent formula `(i - 1) / width`, so it is O(depth) with no
//! allocation.
//!
//! For attention masking, the rule is: `mask[r][c] == 1` iff `c` (the key)
//! is an ancestor-or-self of `r` (the query) in the tree. This is the direct
//! tree-attention analogue of standard causal masking (`mask[r][c] == 1` iff
//! `r >= c` in 1D): a token at position `r` attends to its tree-ancestors,
//! which include itself and everything earlier on its branch.
//!
//! # Example
//!
//! ```text
//! width = 2, depth = 2
//!
//! in-tree indices:        absolute positions (offset = 4):
//!
//!             0                       4
//!           /   \                    / \
//!          1     2                  5   6
//!         / \                       /
//!        3   4                     7
//!
//! is_ancestor_or_self(0, 3) = true   (root is ancestor of 3)
//! is_ancestor_or_self(1, 3) = true   (parent of 3)
//! is_ancestor_or_self(3, 1) = false  (3 is descendant of 1, not ancestor)
//! is_ancestor_or_self(1, 2) = false  (siblings, neither ancestor of other)
//! is_ancestor_or_self(2, 3) = false  (2 is uncle, not ancestor of 3)
//! ```

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreePlan {
    pub width: usize,
    pub depth: usize,
    pub root: Vec<usize>,
}

impl TreePlan {
    pub fn new(width: usize, depth: usize) -> Self {
        Self {
            width,
            depth,
            root: Vec::new(),
        }
    }

    /// Total nodes in the explicit tree (full expansion).
    ///
    /// Returns `1 + width + width^2 + ... + width^depth`. For `depth == 0`
    /// the tree is a single root, so the result is `1` regardless of width.
    pub fn total_nodes(&self) -> usize {
        let mut n = 1usize;
        let mut level = self.width;
        for _ in 0..self.depth {
            n += level;
            level = level.saturating_mul(self.width);
        }
        n
    }

    /// Compute the parent index of node `i`. Returns `None` for the root.
    ///
    /// `i` must be an **in-tree index** (`i == 0` is the root). Callers are
    /// responsible for converting from absolute sequence positions using the
    /// tree's `offset`.
    pub fn parent(i: usize, width: usize) -> Option<usize> {
        if i == 0 || width == 0 {
            None
        } else {
            (i - 1).checked_div(width)
        }
    }
}

/// Convert an absolute sequence position to an in-tree index, if it lies in
/// the tree region `[offset, offset + total_tree_nodes)`.
fn in_tree_index(pos: usize, offset: usize, total_tree_nodes: usize) -> Option<usize> {
    if pos < offset {
        return None;
    }
    let idx = pos - offset;
    if idx < total_tree_nodes {
        Some(idx)
    } else {
        None
    }
}

/// Return `true` iff `ancestor` is the same node as `descendant` or an
/// ancestor of `descendant` in a width-`width` tree.
///
/// Both arguments must be in-tree indices (`0` is the root). The algorithm
/// climbs from `descendant` toward the root using the closed-form parent
/// `(i - 1) / width` and stops as soon as it either finds `ancestor` or
/// drops below it. Because every climb strictly decreases the in-tree
/// index, dropping below `ancestor` is sufficient to conclude that
/// `ancestor` is not on the parent chain.
fn is_ancestor_or_self(ancestor: usize, descendant: usize, width: usize) -> bool {
    let mut cur = descendant;
    loop {
        if cur == ancestor {
            return true;
        }
        if cur < ancestor {
            return false;
        }
        match TreePlan::parent(cur, width) {
            Some(p) => cur = p,
            None => return false,
        }
    }
}

/// Build a [0/1] block-diagonal causal mask for a tree-shaped draft.
///
/// * `mask[r][c] == 1` iff query position `r` may attend to key position
///   `c` in the full sequence.
/// * `seq_len` is the full sequence length (prefix + draft).
/// * `tree_width` is the branching factor; `tree_width >= 1`.
/// * `tree_depth` is the number of levels below the root; `tree_depth == 0`
///   is a degenerate tree of just the root.
/// * `offset` is the absolute sequence position of the tree root.
///
/// The tree occupies `[offset, offset + TreePlan::total_nodes())`. Positions
/// outside that window are in the prefix and obey standard causal masking:
/// `mask[r][c] = (r >= c) ? 1 : 0`.
pub fn tree_causal_mask(
    seq_len: usize,
    tree_width: usize,
    tree_depth: usize,
    offset: usize,
) -> Vec<Vec<u8>> {
    let mut mask = vec![vec![0u8; seq_len]; seq_len];
    let plan = TreePlan::new(tree_width, tree_depth);
    let total_tree_nodes = plan.total_nodes();

    for (r, row) in mask.iter_mut().enumerate() {
        let r_in_tree = in_tree_index(r, offset, total_tree_nodes);
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = match (r_in_tree, in_tree_index(c, offset, total_tree_nodes)) {
                // Both query and key are in the tree. The query can attend
                // to the key iff the key is an ancestor-or-self of the query
                // in the explicit tree.
                (Some(r_idx), Some(c_idx)) => {
                    if is_ancestor_or_self(c_idx, r_idx, tree_width) {
                        1
                    } else {
                        0
                    }
                }
                // Query is in the tree; key is in the prefix. Tree nodes
                // always lie at or after `offset`, so a tree node can
                // attend to any prefix key at or before it.
                (Some(_), None) => {
                    if r >= c {
                        1
                    } else {
                        0
                    }
                }
                // Query is in the prefix; key is in the tree. Prefix queries
                // never see tree keys.
                (None, Some(_)) => 0,
                // Both in the prefix: standard causal.
                (None, None) => {
                    if r >= c {
                        1
                    } else {
                        0
                    }
                }
            };
        }
    }
    mask
}

/// Sparse representation of the tree causal mask.
///
/// Stores only the `(row, col)` pairs where `mask[row][col] == 1`.
/// For a tree of width `W` and depth `D` with prefix `P`, this is
/// `O(P*D*W + P²)` entries instead of `O((P + D*W)²)` for the dense form.
#[derive(Debug, Clone)]
pub struct SparseTreeMask {
    pub seq_len: usize,
    pub entries: Vec<(usize, usize)>,
}

impl SparseTreeMask {
    /// Convert to dense `Vec<Vec<u8>>` for verification or Metal kernel fallback.
    pub fn to_dense(&self) -> Vec<Vec<u8>> {
        let mut mask = vec![vec![0u8; self.seq_len]; self.seq_len];
        for &(r, c) in &self.entries {
            mask[r][c] = 1;
        }
        mask
    }

    /// Number of non-zero entries (useful for benchmarking).
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }
}

/// Build a sparse tree causal mask that stores only the set entries.
///
/// The rule is identical to [`tree_causal_mask`]: `mask[r][c] == 1` iff
/// query position `r` may attend to key position `c`. Positions in the
/// prefix obey standard causal (`r >= c`); positions in the tree obey
/// ancestor-or-self via [`TreePlan`].
pub fn tree_causal_mask_sparse(
    seq_len: usize,
    tree_width: usize,
    tree_depth: usize,
    offset: usize,
) -> SparseTreeMask {
    let plan = TreePlan::new(tree_width, tree_depth);
    let total_tree_nodes = plan.total_nodes();
    let mut entries = Vec::new();

    for r in 0..seq_len {
        let r_in_tree = in_tree_index(r, offset, total_tree_nodes);
        for c in 0..seq_len {
            let c_in_tree = in_tree_index(c, offset, total_tree_nodes);
            let allowed = match (r_in_tree, c_in_tree) {
                (Some(r_idx), Some(c_idx)) => is_ancestor_or_self(c_idx, r_idx, tree_width),
                (Some(_), None) => r >= c,
                (None, Some(_)) => false,
                (None, None) => r >= c,
            };
            if allowed {
                entries.push((r, c));
            }
        }
    }

    SparseTreeMask { seq_len, entries }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_plan_total_nodes() {
        // depth=2 width=4 -> 1 + 4 + 16 = 21
        assert_eq!(TreePlan::new(4, 2).total_nodes(), 21);
        // depth=0 -> just the root
        assert_eq!(TreePlan::new(4, 0).total_nodes(), 1);
        assert_eq!(TreePlan::new(1, 0).total_nodes(), 1);
        // depth=1 width=4 -> 1 + 4 = 5
        assert_eq!(TreePlan::new(4, 1).total_nodes(), 5);
        // width=1 depth=k -> chain of length k+1
        assert_eq!(TreePlan::new(1, 3).total_nodes(), 4);
        // width=2 depth=2 -> 1 + 2 + 4 = 7
        assert_eq!(TreePlan::new(2, 2).total_nodes(), 7);
    }

    #[test]
    fn sparse_mask_matches_dense() {
        for (seq_len, width, depth, offset) in
            [(8, 2, 2, 0), (16, 4, 3, 0), (12, 2, 3, 4), (32, 4, 4, 8)]
        {
            let dense = tree_causal_mask(seq_len, width, depth, offset);
            let sparse = tree_causal_mask_sparse(seq_len, width, depth, offset);
            let sparse_dense = sparse.to_dense();
            assert_eq!(
                dense, sparse_dense,
                "Mismatch at ({seq_len}, {width}, {depth}, {offset})"
            );
        }
    }

    #[test]
    fn sparse_mask_fewer_entries_than_dense() {
        let seq_len = 16;
        let dense = tree_causal_mask(seq_len, 4, 3, 0);
        let sparse = tree_causal_mask_sparse(seq_len, 4, 3, 0);
        let dense_nnz = dense.iter().flatten().filter(|&&x| x == 1).count();
        assert_eq!(sparse.nnz(), dense_nnz);
    }

    #[test]
    fn sparse_mask_memory_comparison() {
        let seq_len = 32;
        let dense = tree_causal_mask(seq_len, 4, 4, 8);
        let sparse = tree_causal_mask_sparse(seq_len, 4, 4, 8);
        let dense_nnz = dense.iter().flatten().filter(|&&x| x == 1).count();
        let dense_cells = seq_len * seq_len;
        eprintln!(
            "seq_len={seq_len}: dense cells={dense_cells}, dense nnz={dense_nnz}, sparse nnz={}",
            sparse.nnz()
        );
        eprintln!(
            "  sparse/dense ratio = {:.2}%",
            sparse.nnz() as f64 / dense_cells as f64 * 100.0
        );
        assert_eq!(sparse.nnz(), dense_nnz);
    }
}
