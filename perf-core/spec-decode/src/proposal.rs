//! Medusa proposal — multiple parallel draft heads on the target model.
//!
//! In Medusa decoding the target model carries several small "heads" attached
//! to the final hidden state. Each head proposes the next one (or two) tokens
//! given the current hidden state; the union of all head proposals is
//! arranged into a candidate tree and verified in a single tree-attention
//! forward pass.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// Tree topology constraints shared by the proposal builder and the verifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeTopology {
    /// Maximum sibling fan-out at every level.
    pub width: usize,
    /// Maximum tree depth (i.e. number of levels, excluding the root anchor).
    pub depth: usize,
}

impl Default for TreeTopology {
    fn default() -> Self {
        Self { width: 4, depth: 1 }
    }
}

/// A Medusa proposal: per-head token lists plus the topology that bound them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MedusaProposal {
    /// `heads[i]` is the top-k token IDs proposed by the i-th head.
    pub heads: Vec<Vec<u32>>,
    /// Topology actually used to bound the proposal.
    pub tree: TreeTopology,
}

impl MedusaProposal {
    /// Build a proposal by polling each head with the running KV cache.
    ///
    /// The result is bounded by `max_tokens`: if the union of all heads
    /// would exceed that cap, earlier heads are kept intact and later heads
    /// are truncated. Within each head, order is preserved.
    pub fn from_heads(
        heads: &[Box<dyn MedusaHead>],
        kv: &[u32],
        tree: &TreeTopology,
        max_tokens: usize,
    ) -> Self {
        let per_head = per_head_budget(max_tokens, tree, heads.len());
        let mut kept: Vec<Vec<u32>> = Vec::with_capacity(heads.len());
        let mut remaining = max_tokens;

        for (i, head) in heads.iter().enumerate() {
            if remaining == 0 {
                break;
            }
            let k = per_head.get(i).copied().unwrap_or(0).min(remaining);
            if k == 0 {
                break;
            }
            let mut tokens = head.propose(kv, k);
            if tokens.len() > k {
                tokens.truncate(k);
            }
            // dedup-preserving-order within head
            tokens = dedup_preserve(tokens);
            if tokens.len() > remaining {
                tokens.truncate(remaining);
            }
            remaining -= tokens.len();
            kept.push(tokens);
        }

        Self {
            heads: kept,
            tree: *tree,
        }
    }

    /// Flattened ordered token list across all heads (used for the verifier
    /// when no tree structure is required).
    pub fn flat_tokens(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for h in &self.heads {
            for &t in h {
                if !out.contains(&t) {
                    out.push(t);
                }
            }
        }
        out
    }

    /// Total proposed tokens across all heads.
    pub fn total(&self) -> usize {
        self.heads.iter().map(|h| h.len()).sum()
    }

    /// Build a proposal that returns no candidates — useful as a degenerate
    /// branch in error handling and cancellation paths.
    pub fn empty(tree: TreeTopology) -> Self {
        Self {
            heads: Vec::new(),
            tree,
        }
    }
}

/// A single Medusa draft head. Implementors receive the live KV cache slice
/// (already-accepted tokens up to and including the previous step) and the
/// per-head budget `k`; they return at most `k` token IDs in order of
/// descending probability.
pub trait MedusaHead {
    fn propose(&self, kv: &[u32], k: usize) -> Vec<u32>;
}

/// Deterministic test fixture: emits from a pre-loaded token table, in order.
#[derive(Debug, Clone)]
pub struct MockMedusaHead {
    table: Vec<u32>,
}

impl MockMedusaHead {
    pub fn new(table: Vec<u32>) -> Self {
        Self { table }
    }
    pub fn table(&self) -> &[u32] {
        &self.table
    }
}

impl MedusaHead for MockMedusaHead {
    fn propose(&self, _kv: &[u32], k: usize) -> Vec<u32> {
        if k == 0 {
            return Vec::new();
        }
        let n = k
            .min(self.table.len())
            .max(if self.table.is_empty() { 0 } else { 1 });
        let take = n.min(self.table.len());
        self.table[..take].to_vec()
    }
}

/// Compute the per-head `k` budget given the global cap and topology.
///
/// The first head gets the largest slice (it is anchored at the prompt and
/// therefore always contributes). Side heads receive progressively smaller
/// budgets down to 1.
fn per_head_budget(total: usize, tree: &TreeTopology, n_heads: usize) -> Vec<usize> {
    if n_heads == 0 || total == 0 {
        return Vec::new();
    }
    // width bounds how many candidates a single level can carry; depth
    // bounds how many tokens of vertical extent the tree has.
    let depth_cap = tree.depth.max(1);
    let width_cap = tree.width.max(1);

    let base = total / n_heads;
    let rem = total % n_heads;
    (0..n_heads)
        .map(|i| {
            let b = base + if i < rem { 1 } else { 0 };
            b.max(1).min(width_cap).min(depth_cap)
        })
        .collect()
}

/// Dedup a slice preserving first-seen order.
pub fn dedup_preserve(xs: Vec<u32>) -> Vec<u32> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(xs.len());
    for x in xs {
        if seen.insert(x) {
            out.push(x);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_preserves_order() {
        let v = vec![3, 1, 3, 2, 1, 4];
        assert_eq!(dedup_preserve(v), vec![3, 1, 2, 4]);
    }

    #[test]
    fn per_head_budget_respects_width_and_depth() {
        let tree = TreeTopology { width: 2, depth: 1 };
        let budget = per_head_budget(10, &tree, 4);
        assert_eq!(budget, vec![1, 1, 1, 1]);
    }

    #[test]
    fn per_head_budget_zero_total() {
        let tree = TreeTopology::default();
        assert!(per_head_budget(0, &tree, 4).is_empty());
    }

    #[test]
    fn from_heads_caps_total_at_max() {
        let heads: Vec<Box<dyn MedusaHead>> = vec![
            Box::new(MockMedusaHead::new(vec![1, 2, 3, 4, 5])),
            Box::new(MockMedusaHead::new(vec![6, 7, 8, 9, 10])),
        ];
        let p = MedusaProposal::from_heads(&heads, &[0], &TreeTopology { width: 4, depth: 2 }, 3);
        assert!(p.total() <= 3, "got {}", p.total());
    }

    #[test]
    fn empty_proposal_has_no_tokens() {
        let p = MedusaProposal::empty(TreeTopology::default());
        assert_eq!(p.total(), 0);
        assert!(p.flat_tokens().is_empty());
    }

    #[test]
    fn flat_tokens_dedups_across_heads() {
        let heads: Vec<Box<dyn MedusaHead>> = vec![
            Box::new(MockMedusaHead::new(vec![1, 2])),
            Box::new(MockMedusaHead::new(vec![2, 3])),
        ];
        let p = MedusaProposal::from_heads(&heads, &[0], &TreeTopology { width: 4, depth: 2 }, 8);
        let flat = p.flat_tokens();
        assert_eq!(flat, vec![1, 2, 3]);
    }

    // -- dedup_preserve edge cases ----------------------------------------------------

    #[test]
    fn dedup_preserve_insertion_order_first_seen_wins() {
        let input = vec![10, 20, 10, 30, 20, 40, 30];
        let result = dedup_preserve(input);
        assert_eq!(result, vec![10, 20, 30, 40]);
    }

    #[test]
    fn dedup_preserve_empty_input_returns_empty() {
        let result = dedup_preserve(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn dedup_preserve_all_duplicates_returns_single_element() {
        let result = dedup_preserve(vec![7, 7, 7, 7]);
        assert_eq!(result, vec![7]);
    }

    #[test]
    fn dedup_preserve_no_duplicates_preserves_all() {
        let input = vec![1, 2, 3, 4, 5];
        let result = dedup_preserve(input.clone());
        assert_eq!(result, input);
    }

    #[test]
    fn dedup_preserve_single_element_returns_same() {
        assert_eq!(dedup_preserve(vec![42]), vec![42]);
    }

    #[test]
    fn dedup_preserve_adjacent_duplicates() {
        let result = dedup_preserve(vec![1, 1, 2, 2, 3, 3]);
        assert_eq!(result, vec![1, 2, 3]);
    }
}
