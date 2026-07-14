//! Tree-attention — explicit causal masks for JetSpec-style draft trees.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreePlan {
    pub width: usize,
    pub depth: usize,
    pub root: Vec<usize>,
}

impl TreePlan {
    pub fn new(width: usize, depth: usize) -> Self {
        Self { width, depth, root: Vec::new() }
    }

    /// Total nodes in the explicit tree (full expansion).
    pub fn total_nodes(&self) -> usize {
        let mut n = 1;
        let mut level = self.width;
        for _ in 1..self.depth.max(1) {
            n += level;
            level *= self.width;
        }
        n
    }

    /// Compute the parent index of node `i`. Returns `None` for the root.
    pub fn parent(i: usize, width: usize) -> Option<usize> {
        if i == 0 { None } else { Some((i - 1) / width) }
    }
}

/// Build a [0/1] block-diagonal causal mask for a tree-shaped draft.
///
/// `seq_len` — full sequence length (prefix + draft).
/// `tree_width` — branching factor.
/// `tree_depth` — number of levels below root.
/// `offset` — start position of the tree in the full sequence.
pub fn tree_causal_mask(seq_len: usize, tree_width: usize, tree_depth: usize, offset: usize) -> Vec<Vec<u8>> {
    let mut mask = vec![vec![0u8; seq_len]; seq_len];
    let tree_start = offset;
    let tree_len = tree_width.pow(tree_depth as u32);
    let tree_end = tree_start + tree_len;

    for r in 0..seq_len {
        for c in 0..seq_len {
            let same_tree = c >= tree_start && c < tree_end;
            if same_tree && r >= tree_start {
                // 1 if r is ancestor-or-self of c.
                let mut cur = c;
                let mut anc = true;
                while cur > r {
                    cur = match TreePlan::parent(cur, tree_width) {
                        Some(p) => p,
                        None => break,
                    };
                    if cur <= r { break; }
                    if cur == r { break; }
                }
                if cur == r || r == c || (c > r && cur < r) {
                    mask[r][c] = 1;
                }
                // Prefix attention always allowed
                if c < tree_start && r >= c { mask[r][c] = 1; }
            } else {
                // Default causal
                mask[r][c] = if c <= r { 1 } else { 0 };
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_plan_total_nodes() {
        let t = TreePlan::new(4, 2);
        assert!(t.total_nodes() >= 5);
    }
}