use async_trait::async_trait;
use spec_decode::backend::{BackendInfo, TargetOutput};
use spec_decode::tree_proposal::{
    create_parallel_trees, merge_parallel_results, ParallelTreeConfig,
};
use spec_decode::{ProposalMode, SpecDecodeConfig, SpecDecodeEngine, TargetBackend};

struct AcceptAllTarget;

#[async_trait]
impl TargetBackend for AcceptAllTarget {
    async fn forward(&self, _: &[u32]) -> Result<TargetOutput, String> {
        let mut logits = vec![0.0_f32; 256];
        logits[0] = 10.0;
        Ok(TargetOutput {
            logits,
            hidden: None,
            finished: false,
        })
    }

    async fn verify_tree(
        &self,
        _: &[u32],
        candidates: &[Vec<u32>],
    ) -> Result<Vec<bool>, String> {
        Ok(vec![true; candidates.len()])
    }

    fn info(&self) -> BackendInfo {
        BackendInfo {
            engine: "test".into(),
            model_id: "accept-all".into(),
            device: "cpu".into(),
            dtype: "f32".into(),
            kv_cache_type: None,
        }
    }
}

struct RejectAllTarget;

#[async_trait]
impl TargetBackend for RejectAllTarget {
    async fn forward(&self, _: &[u32]) -> Result<TargetOutput, String> {
        let mut logits = vec![0.0_f32; 256];
        logits[42] = 10.0;
        Ok(TargetOutput {
            logits,
            hidden: None,
            finished: false,
        })
    }

    async fn verify_tree(
        &self,
        _: &[u32],
        candidates: &[Vec<u32>],
    ) -> Result<Vec<bool>, String> {
        Ok(vec![false; candidates.len()])
    }

    fn info(&self) -> BackendInfo {
        BackendInfo {
            engine: "test".into(),
            model_id: "reject-all".into(),
            device: "cpu".into(),
            dtype: "f32".into(),
            kv_cache_type: None,
        }
    }
}

#[tokio::test]
async fn e2e_eagle3_full_step_cycle() {
    let config = ParallelTreeConfig {
        num_parallel_branches: 4,
        max_depth: 3,
        max_branches_per_node: 3,
        probability_threshold: 0.01,
    };

    let mut engine = SpecDecodeEngine::new(
        SpecDecodeConfig::default(),
        Box::new(AcceptAllTarget),
        None,
    )
    .with_proposal_mode(ProposalMode::Eagle3(config));

    for step in 0..5 {
        let prefix = vec![100 + step as u32];

        let branch_logits = vec![
            vec![(201, 0.5), (202, 0.3), (203, 0.2)],
            vec![(301, 0.4), (302, 0.4), (303, 0.2)],
            vec![(401, 0.6), (402, 0.3), (403, 0.1)],
        ];

        let result = engine.step_eagle3(&prefix, branch_logits).await;
        assert!(result.is_ok(), "Step {} failed: {:?}", step, result.err());

        let step_result = result.unwrap();
        assert!(
            !step_result.accepted.is_empty() || step == 0,
            "Step {} should have accepted tokens",
            step
        );

        for tok in &step_result.accepted {
            assert!(tok.was_drafted, "Accepted tokens should be drafted");
        }
    }
}

#[tokio::test]
async fn e2e_eagle3_reject_all_returns_empty() {
    let config = ParallelTreeConfig {
        num_parallel_branches: 2,
        max_depth: 2,
        max_branches_per_node: 2,
        probability_threshold: 0.01,
    };

    let mut engine = SpecDecodeEngine::new(
        SpecDecodeConfig::default(),
        Box::new(RejectAllTarget),
        None,
    )
    .with_proposal_mode(ProposalMode::Eagle3(config));

    let branch_logits = vec![
        vec![(10, 0.6), (11, 0.3), (12, 0.1)],
        vec![(20, 0.5), (21, 0.5)],
    ];

    let result = engine.step_eagle3(&[1, 2, 3], branch_logits).await;
    assert!(result.is_ok());

    let step_result = result.unwrap();
    assert!(
        step_result.accepted.is_empty(),
        "step_eagle3 returns empty accepted when all candidates rejected"
    );
    assert!(step_result.drafted > 0, "Should have drafted candidates");
}

#[tokio::test]
async fn e2e_eagle3_state_accumulates_across_steps() {
    let mut engine = SpecDecodeEngine::new(
        SpecDecodeConfig::default(),
        Box::new(AcceptAllTarget),
        None,
    )
    .with_proposal_mode(ProposalMode::Eagle3(ParallelTreeConfig {
        num_parallel_branches: 2,
        max_depth: 2,
        max_branches_per_node: 2,
        probability_threshold: 0.01,
    }));

    let mut total_accepted: usize = 0;
    for step in 0..3 {
        let branch_logits = vec![
            vec![
                (500 + step as u32, 0.6),
                (600 + step as u32, 0.3),
                (700 + step as u32, 0.05),
                (800 + step as u32, 0.05),
            ],
        ];
        let result = engine
            .step_eagle3(&[step as u32], branch_logits)
            .await
            .unwrap();
        total_accepted += result.accepted.len();
    }

    let state = engine.state();
    assert_eq!(state.accepted_total, total_accepted as u64);
    assert!(state.kv_len > 0, "KV should have grown from steps");
}

#[test]
fn e2e_eagle3_tree_stats() {
    let branch_logits = vec![
        vec![
            (1, 0.5), (2, 0.3), (3, 0.2),
            (4, 0.45), (5, 0.35), (6, 0.2),
            (7, 0.4), (8, 0.35), (9, 0.25),
        ],
        vec![(10, 0.6), (11, 0.3), (12, 0.1)],
    ];

    let config = ParallelTreeConfig {
        num_parallel_branches: 4,
        max_depth: 3,
        max_branches_per_node: 3,
        probability_threshold: 0.01,
    };

    let trees = create_parallel_trees(0, branch_logits.clone(), &config);
    assert_eq!(trees.len(), 3, "9 first-level candidates / 3 per branch = 3 trees");

    for (i, tree) in trees.iter().enumerate() {
        assert!(
            tree.node_count() > 0,
            "Tree {} should have nodes",
            i
        );
        assert!(
            tree.depth > 0 || tree.root.children.is_empty(),
            "Tree {} should have depth > 0 or be single node",
            i
        );
    }

    let best = merge_parallel_results(&trees);
    assert!(!best.is_empty(), "Merged result should have tokens");
}

#[test]
fn e2e_eagle3_tree_leaf_paths_match_expected_count() {
    let branch_logits = vec![
        vec![(10, 0.5), (20, 0.3)],
        vec![(30, 0.6), (40, 0.4)],
    ];
    let tree = spec_decode::DraftTree::from_eagle3_predictions(0, branch_logits, 4, 3);
    let paths = tree.leaf_paths();
    assert_eq!(paths.len(), 4);
    for p in &paths {
        assert_eq!(p.len(), 2);
    }
}

#[tokio::test]
async fn e2e_eagle3_vs_medusa_same_input() {
    let mut eagle = SpecDecodeEngine::new(
        SpecDecodeConfig::default(),
        Box::new(AcceptAllTarget),
        None,
    )
    .with_proposal_mode(ProposalMode::Eagle3(ParallelTreeConfig::default()));

    let branch_logits = vec![vec![(10, 0.6), (11, 0.3), (12, 0.1)]];

    let eagle_result = eagle.step_eagle3(&[5], branch_logits).await;
    assert!(eagle_result.is_ok(), "EAGLE-3 should not panic");

    let mut medusa = SpecDecodeEngine::new(
        SpecDecodeConfig::default(),
        Box::new(AcceptAllTarget),
        None,
    );

    let medusa_result = medusa.step(&[5]).await;
    assert!(medusa_result.is_ok(), "Medusa legacy path should not panic");
}

#[test]
fn e2e_eagle3_config_round_trip() {
    let config = ParallelTreeConfig {
        num_parallel_branches: 8,
        max_depth: 16,
        max_branches_per_node: 4,
        probability_threshold: 0.001,
    };

    let engine = SpecDecodeEngine::new(
        SpecDecodeConfig::default(),
        Box::new(AcceptAllTarget),
        None,
    )
    .with_proposal_mode(ProposalMode::Eagle3(config.clone()));

    match &engine.proposal_mode {
        ProposalMode::Eagle3(cfg) => {
            assert_eq!(cfg.num_parallel_branches, 8);
            assert_eq!(cfg.max_depth, 16);
            assert_eq!(cfg.max_branches_per_node, 4);
            assert_eq!(cfg.probability_threshold, 0.001);
        }
        _ => panic!("expected Eagle3 variant"),
    }
}
