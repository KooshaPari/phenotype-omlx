//! EAGLE-3 and P-EAGLE tree-based draft proposals.
//!
//! EAGLE-3 proposes entire draft subtrees by predicting multiple candidate
//! continuations at each decoding depth. The tree is traversed depth-first
//! during verification; the first valid continuation is accepted.
//!
//! P-EAGLE extends this by running multiple draft branches in parallel,
//! performing speculative verification across all branches simultaneously,
//! and selecting the best valid sequence.

use serde::{Deserialize, Serialize};

/// Admission limits for speculative tree construction. These are deliberately
/// conservative process-wide caps: untrusted/request-derived configuration
/// must not turn one decoding step into an exponential allocation.
pub const MAX_PARALLEL_BRANCHES: usize = 64;
pub const MAX_TREE_DEPTH: usize = 64;
pub const MAX_BRANCHES_PER_NODE: usize = 32;

/// A node in the draft tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftNode {
    pub token_id: u32,
    pub probability: f32,
    pub children: Vec<DraftNode>,
}

/// A complete draft tree produced by EAGLE-3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftTree {
    pub root: DraftNode,
    pub depth: usize,
    pub total_leaves: usize,
}

impl DraftTree {
    /// Build a tree from EAGLE-3 style predictions.
    ///
    /// `root_token` is the anchor token (e.g. last accepted token).
    /// `branch_logits` is a `Vec` of per-depth candidate lists where each
    /// entry is `Vec<(token_id, probability)>` sorted descending by
    /// probability.
    /// `max_depth` bounds the tree height; `max_branches` bounds the fan-out
    /// at each internal node.
    pub fn from_eagle3_predictions(
        root_token: u32,
        branch_logits: Vec<Vec<(u32, f32)>>,
        max_depth: usize,
        max_branches: usize,
    ) -> Self {
        let root = DraftNode {
            token_id: root_token,
            probability: 1.0,
            children: Vec::new(),
        };

        let max_depth = max_depth.min(MAX_TREE_DEPTH);
        let max_branches = max_branches.min(MAX_BRANCHES_PER_NODE);
        if branch_logits.is_empty() || max_depth == 0 || max_branches == 0 {
            return Self {
                root,
                depth: 0,
                total_leaves: 1,
            };
        }

        let mut tree = Self {
            root,
            depth: 0,
            total_leaves: 1,
        };

        for (depth_idx, candidates) in branch_logits.iter().enumerate() {
            if depth_idx >= max_depth {
                break;
            }
            let pruned: Vec<(u32, f32)> = candidates.iter().take(max_branches).cloned().collect();

            if pruned.is_empty() {
                break;
            }

            let new_leaves = Self::attach_level(&mut tree.root, &pruned, depth_idx, 0);
            tree.depth = depth_idx + 1;
            tree.total_leaves = new_leaves;
        }

        tree
    }

    /// Recursively attach children at the given depth level.
    fn attach_level(
        node: &mut DraftNode,
        candidates: &[(u32, f32)],
        target_depth: usize,
        current_depth: usize,
    ) -> usize {
        if current_depth == target_depth {
            node.children = candidates
                .iter()
                .map(|&(tok, prob)| DraftNode {
                    token_id: tok,
                    probability: prob,
                    children: Vec::new(),
                })
                .collect();
            node.children.len()
        } else {
            let mut leaves = 0;
            for child in &mut node.children {
                leaves += Self::attach_level(child, candidates, target_depth, current_depth + 1);
            }
            if leaves == 0 {
                leaves = 1;
            }
            leaves
        }
    }

    /// Return every root-to-leaf path as a token sequence (excluding the
    /// root token itself).
    pub fn leaf_paths(&self) -> Vec<Vec<u32>> {
        let mut paths = Vec::new();
        Self::collect_paths(&self.root, &mut Vec::new(), &mut paths);
        paths
    }

    fn collect_paths(node: &DraftNode, prefix: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
        if node.children.is_empty() {
            out.push(prefix.clone());
            return;
        }
        for child in &node.children {
            prefix.push(child.token_id);
            Self::collect_paths(child, prefix, out);
            prefix.pop();
        }
    }

    /// Return the path with the highest cumulative probability product.
    pub fn best_path(&self) -> Vec<u32> {
        let mut best: Option<(f32, Vec<u32>)> = None;
        Self::find_best(&self.root, 1.0, &mut Vec::new(), &mut best);
        best.map(|(_, p)| p).unwrap_or_default()
    }

    fn find_best(
        node: &DraftNode,
        cum_prob: f32,
        path: &mut Vec<u32>,
        best: &mut Option<(f32, Vec<u32>)>,
    ) {
        if node.children.is_empty() {
            match best {
                None => *best = Some((cum_prob, path.clone())),
                Some((bp, _)) if cum_prob > *bp => {
                    *best = Some((cum_prob, path.clone()));
                }
                _ => {}
            }
            return;
        }
        for child in &node.children {
            path.push(child.token_id);
            Self::find_best(child, cum_prob * child.probability, path, best);
            path.pop();
        }
    }

    /// Total number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        Self::count_nodes(&self.root)
    }

    fn count_nodes(node: &DraftNode) -> usize {
        1 + node.children.iter().map(Self::count_nodes).sum::<usize>()
    }

    /// Remove every child whose probability is below `threshold`.
    pub fn prune(&mut self, threshold: f32) {
        Self::prune_node(&mut self.root, threshold);
        self.total_leaves = self.count_leaves(&self.root);
    }

    fn prune_node(node: &mut DraftNode, threshold: f32) {
        node.children.retain(|c| c.probability >= threshold);
        for child in &mut node.children {
            Self::prune_node(child, threshold);
        }
    }

    fn count_leaves(&self, node: &DraftNode) -> usize {
        if node.children.is_empty() {
            1
        } else {
            node.children.iter().map(|c| self.count_leaves(c)).sum()
        }
    }
}

/// Configuration for P-EAGLE parallel tree exploration.
#[derive(Debug, Clone)]
pub struct ParallelTreeConfig {
    /// Number of distinct draft trees to explore in parallel.
    pub num_parallel_branches: usize,
    /// Maximum tree depth for each branch.
    pub max_depth: usize,
    /// Maximum child fan-out per node.
    pub max_branches_per_node: usize,
    /// Probability below which a node is pruned.
    pub probability_threshold: f32,
}

impl Default for ParallelTreeConfig {
    fn default() -> Self {
        Self {
            num_parallel_branches: 4,
            max_depth: 8,
            max_branches_per_node: 3,
            probability_threshold: 0.01,
        }
    }
}

/// Build `num_parallel_branches` draft trees from the shared logits.
///
/// Each tree receives a distinct subset of candidates at its root level to
/// encourage diversity. Deeper levels share the same logits.
pub fn create_parallel_trees(
    root_token: u32,
    branch_logits: Vec<Vec<(u32, f32)>>,
    config: &ParallelTreeConfig,
) -> Vec<DraftTree> {
    if branch_logits.is_empty() {
        return vec![DraftTree::from_eagle3_predictions(
            root_token,
            Vec::new(),
            config.max_depth,
            config.max_branches_per_node,
        )];
    }

    let first_level = &branch_logits[0];
    let n_branches = config
        .num_parallel_branches
        .clamp(1, MAX_PARALLEL_BRANCHES);
    let max_depth = config.max_depth.min(MAX_TREE_DEPTH);
    let max_branches = config
        .max_branches_per_node
        .min(MAX_BRANCHES_PER_NODE);
    let mut trees = Vec::with_capacity(n_branches);

    for b in 0..n_branches {
        let start = b.saturating_mul(max_branches);
        if start >= first_level.len() {
            break;
        }
        let end = start.saturating_add(max_branches).min(first_level.len());
        let slice: Vec<(u32, f32)> = first_level[start..end].to_vec();

        if slice.is_empty() {
            break;
        }

        let mut logits = branch_logits.clone();
        logits[0] = slice;

        let tree = DraftTree::from_eagle3_predictions(
            root_token,
            logits,
            max_depth,
            max_branches,
        );
        trees.push(tree);
    }

    if trees.is_empty() {
        trees.push(DraftTree::from_eagle3_predictions(
            root_token,
            Vec::new(),
            max_depth,
            max_branches,
        ));
    }

    trees
}

/// Merge results from multiple parallel trees by selecting the single best
/// leaf path across all trees.
pub fn merge_parallel_results(trees: &[DraftTree]) -> Vec<u32> {
    let mut best: Option<(f32, Vec<u32>)> = None;

    for tree in trees {
        let path = tree.best_path();
        let prob = path_probability(&tree.root, &path);

        match &best {
            None => best = Some((prob, path)),
            Some((bp, _)) if prob > *bp => {
                best = Some((prob, path));
            }
            _ => {}
        }
    }

    best.map(|(_, p)| p).unwrap_or_default()
}

/// Compute cumulative probability for a given path through the tree.
fn path_probability(root: &DraftNode, path: &[u32]) -> f32 {
    let mut current = root;
    let mut prob = 1.0_f32;

    for &tok in path {
        match current.children.iter().find(|c| c.token_id == tok) {
            Some(child) => {
                prob *= child.probability;
                current = child;
            }
            None => return 0.0,
        }
    }

    prob
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_token_no_branching() {
        let tree = DraftTree::from_eagle3_predictions(10, vec![], 8, 3);
        assert_eq!(tree.root.token_id, 10);
        assert_eq!(tree.depth, 0);
        assert_eq!(tree.total_leaves, 1);
        assert_eq!(tree.node_count(), 1);
        assert_eq!(tree.leaf_paths(), vec![Vec::<u32>::new()]);
        assert_eq!(tree.best_path(), Vec::<u32>::new());
    }

    #[test]
    fn empty_logits_produces_single_node() {
        let tree = DraftTree::from_eagle3_predictions(5, Vec::new(), 4, 2);
        assert_eq!(tree.node_count(), 1);
        assert_eq!(tree.depth, 0);
    }

    #[test]
    fn from_eagle3_predictions_builds_correct_depth() {
        let logits = vec![vec![(1, 0.6), (2, 0.3)], vec![(10, 0.5), (11, 0.4)]];
        let tree = DraftTree::from_eagle3_predictions(0, logits, 4, 3);
        assert_eq!(tree.depth, 2);
        assert_eq!(tree.root.token_id, 0);
        assert_eq!(tree.root.children.len(), 2);
    }

    #[test]
    fn from_eagle3_predictions_respects_max_branches() {
        let logits = vec![vec![(1, 0.4), (2, 0.3), (3, 0.2), (4, 0.1)]];
        let tree = DraftTree::from_eagle3_predictions(0, logits, 4, 2);
        assert_eq!(tree.root.children.len(), 2);
    }

    #[test]
    fn leaf_paths_returns_correct_count() {
        let logits = vec![vec![(1, 0.5), (2, 0.5)], vec![(10, 0.5), (11, 0.5)]];
        let tree = DraftTree::from_eagle3_predictions(0, logits, 4, 3);
        let paths = tree.leaf_paths();
        assert_eq!(paths.len(), 4);
        for p in &paths {
            assert_eq!(p.len(), 2);
        }
    }

    #[test]
    fn best_path_returns_highest_probability() {
        let logits = vec![vec![(1, 0.9), (2, 0.1)], vec![(10, 0.8), (11, 0.2)]];
        let tree = DraftTree::from_eagle3_predictions(0, logits, 4, 3);
        let best = tree.best_path();
        assert_eq!(best, vec![1, 10]);
    }

    #[test]
    fn prune_removes_low_probability_branches() {
        let logits = vec![vec![(1, 0.9), (2, 0.005)]];
        let mut tree = DraftTree::from_eagle3_predictions(0, logits, 4, 3);
        tree.prune(0.01);
        assert_eq!(tree.root.children.len(), 1);
        assert_eq!(tree.root.children[0].token_id, 1);
        assert_eq!(tree.total_leaves, 1);
    }

    #[test]
    fn prune_empty_tree_no_panic() {
        let mut tree = DraftTree::from_eagle3_predictions(0, vec![], 4, 3);
        tree.prune(0.5);
        assert_eq!(tree.total_leaves, 1);
    }

    #[test]
    fn create_parallel_trees_returns_correct_count() {
        let logits = vec![vec![(1, 0.4), (2, 0.3), (3, 0.2), (4, 0.1)]];
        let config = ParallelTreeConfig {
            num_parallel_branches: 2,
            max_depth: 4,
            max_branches_per_node: 2,
            probability_threshold: 0.01,
        };
        let trees = create_parallel_trees(0, logits, &config);
        assert_eq!(trees.len(), 2);
        assert_eq!(trees[0].root.children[0].token_id, 1);
        assert_eq!(trees[1].root.children[0].token_id, 3);
    }

    #[test]
    fn create_parallel_trees_fewer_candidates_than_branches() {
        let logits = vec![vec![(1, 0.6), (2, 0.4)]];
        let config = ParallelTreeConfig {
            num_parallel_branches: 4,
            max_depth: 4,
            max_branches_per_node: 2,
            probability_threshold: 0.01,
        };
        let trees = create_parallel_trees(0, logits, &config);
        assert_eq!(trees.len(), 1);
    }

    #[test]
    fn merge_parallel_results_selects_best() {
        let logits = vec![vec![(1, 0.9), (2, 0.1)]];
        let config = ParallelTreeConfig {
            num_parallel_branches: 2,
            max_depth: 1,
            max_branches_per_node: 1,
            probability_threshold: 0.01,
        };
        let trees = create_parallel_trees(0, logits, &config);
        let merged = merge_parallel_results(&trees);
        assert_eq!(merged, vec![1]);
    }

    #[test]
    fn merge_empty_trees_returns_empty() {
        let merged = merge_parallel_results(&[]);
        assert!(merged.is_empty());
    }

    #[test]
    fn node_count_matches_actual_nodes() {
        let logits = vec![vec![(1, 0.5), (2, 0.5)], vec![(10, 1.0)]];
        let tree = DraftTree::from_eagle3_predictions(0, logits, 4, 3);
        // root(0) + children(1,2) + grandchildren(10,10) = 5
        assert_eq!(tree.node_count(), 5);
    }

    #[test]
    fn parallel_trees_empty_logits() {
        let config = ParallelTreeConfig {
            num_parallel_branches: 4,
            max_depth: 8,
            max_branches_per_node: 3,
            probability_threshold: 0.01,
        };
        let trees = create_parallel_trees(0, Vec::new(), &config);
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].depth, 0);
    }

    #[test]
    fn parallel_tree_limits_cap_request_derived_fanout() {
        let logits = vec![vec![(1, 1.0); 1_000]];
        let config = ParallelTreeConfig {
            num_parallel_branches: usize::MAX,
            max_depth: usize::MAX,
            max_branches_per_node: usize::MAX,
            probability_threshold: 0.0,
        };
        let trees = create_parallel_trees(0, logits, &config);
        assert!(trees.len() <= MAX_PARALLEL_BRANCHES);
        assert!(trees[0].depth <= MAX_TREE_DEPTH);
        assert!(trees[0].root.children.len() <= MAX_BRANCHES_PER_NODE);
    }
}
