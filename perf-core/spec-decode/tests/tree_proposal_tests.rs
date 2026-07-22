use spec_decode::tree_proposal::{
    create_parallel_trees, merge_parallel_results, DraftTree, ParallelTreeConfig,
};

#[test]
fn from_eagle3_predictions_builds_tree_from_known_logits() {
    let logits = vec![vec![(10, 0.6), (20, 0.3)], vec![(30, 0.5), (40, 0.4)]];
    let tree = DraftTree::from_eagle3_predictions(1, logits, 4, 3);
    assert_eq!(tree.root.token_id, 1);
    assert_eq!(tree.depth, 2);
    assert_eq!(tree.root.children.len(), 2);
    assert_eq!(tree.root.children[0].token_id, 10);
    assert_eq!(tree.root.children[1].token_id, 20);
}

#[test]
fn leaf_paths_returns_all_combinations() {
    let logits = vec![vec![(1, 0.5), (2, 0.5)], vec![(10, 0.5), (11, 0.5)]];
    let tree = DraftTree::from_eagle3_predictions(0, logits, 4, 3);
    let paths = tree.leaf_paths();
    assert_eq!(paths.len(), 4);
    assert!(paths.contains(&vec![1, 10]));
    assert!(paths.contains(&vec![1, 11]));
    assert!(paths.contains(&vec![2, 10]));
    assert!(paths.contains(&vec![2, 11]));
}

#[test]
fn best_path_returns_highest_probability_sequence() {
    let logits = vec![vec![(1, 0.9), (2, 0.1)], vec![(10, 0.8), (11, 0.2)]];
    let tree = DraftTree::from_eagle3_predictions(0, logits, 4, 3);
    let best = tree.best_path();
    assert_eq!(best, vec![1, 10]);
}

#[test]
fn prune_removes_low_probability_branches() {
    let logits = vec![vec![(1, 0.9), (2, 0.005)], vec![(10, 0.5), (11, 0.5)]];
    let mut tree = DraftTree::from_eagle3_predictions(0, logits, 4, 3);
    tree.prune(0.01);
    assert_eq!(tree.root.children.len(), 1);
    assert_eq!(tree.root.children[0].token_id, 1);
    assert_eq!(tree.total_leaves, 2);
}

#[test]
fn create_parallel_trees_produces_distinct_trees() {
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
fn merge_parallel_results_picks_best_across_trees() {
    let logits = vec![vec![(10, 0.8), (20, 0.2), (30, 0.9), (40, 0.1)]];
    let config = ParallelTreeConfig {
        num_parallel_branches: 2,
        max_depth: 1,
        max_branches_per_node: 2,
        probability_threshold: 0.01,
    };
    let trees = create_parallel_trees(0, logits, &config);
    assert_eq!(trees.len(), 2);
    let merged = merge_parallel_results(&trees);
    assert_eq!(merged, vec![30]);
}

#[test]
fn single_token_no_branching() {
    let tree = DraftTree::from_eagle3_predictions(42, vec![], 8, 3);
    assert_eq!(tree.root.token_id, 42);
    assert_eq!(tree.depth, 0);
    assert_eq!(tree.total_leaves, 1);
    assert!(tree.leaf_paths().contains(&Vec::<u32>::new()));
    assert!(tree.best_path().is_empty());
}

#[test]
fn empty_logits_produces_single_node() {
    let tree = DraftTree::from_eagle3_predictions(5, Vec::new(), 4, 2);
    assert_eq!(tree.node_count(), 1);
    assert_eq!(tree.depth, 0);
    assert_eq!(tree.total_leaves, 1);
}

#[test]
fn from_eagle3_predictions_respects_max_depth() {
    let logits = vec![vec![(1, 0.5)], vec![(2, 0.5)], vec![(3, 0.5)]];
    let tree = DraftTree::from_eagle3_predictions(0, logits, 2, 3);
    assert_eq!(tree.depth, 2);
}

#[test]
fn from_eagle3_predictions_respects_max_branches() {
    let logits = vec![vec![(1, 0.3), (2, 0.25), (3, 0.25), (4, 0.2)]];
    let tree = DraftTree::from_eagle3_predictions(0, logits, 4, 2);
    assert_eq!(tree.root.children.len(), 2);
}

#[test]
fn parallel_trees_fallback_when_no_candidates() {
    let config = ParallelTreeConfig::default();
    let trees = create_parallel_trees(0, Vec::new(), &config);
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].depth, 0);
}

#[test]
fn merge_empty_slice_returns_empty() {
    let merged = merge_parallel_results(&[]);
    assert!(merged.is_empty());
}
