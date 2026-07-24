use super::*;

// -----------------------------------------------------------------------------
// Medusa proposal contract
// -----------------------------------------------------------------------------

#[test]
fn engine_step_medusa_collects_candidates_from_each_head() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let config = SpecDecodeConfig {
            mode: DraftMode::Medusa,
            max_draft_tokens: 6,
            ..Default::default()
        };

        let mut engine =
            SpecDecodeEngine::new(config, Box::new(ScriptedTarget::with_argmax(64, 7)), None);

        let heads: Vec<Box<dyn MedusaHead>> = vec![
            Box::new(MockMedusaHead::new(vec![11, 12])),
            Box::new(MockMedusaHead::new(vec![21, 22])),
            Box::new(MockMedusaHead::new(vec![31, 32])),
        ];

        let r = engine.step_medusa(&[1, 2, 3], &heads).await.unwrap();
        assert!(r.drafted > 0, "expected drafted > 0, got {}", r.drafted);
    });
}

#[test]
fn engine_step_medusa_deduplicates_by_token_id_preserving_order() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let config = SpecDecodeConfig {
            mode: DraftMode::Medusa,
            max_draft_tokens: 8,
            ..Default::default()
        };

        let heads: Vec<Box<dyn MedusaHead>> = vec![
            Box::new(MockMedusaHead::new(vec![11, 12])),
            Box::new(MockMedusaHead::new(vec![11, 12])),
        ];

        let prop = MedusaProposal::from_heads(
            &heads,
            &[1, 2, 3],
            &TreeTopology { width: 4, depth: 2 },
            config.max_draft_tokens,
        );
        assert_eq!(prop.heads.len(), 2);
        assert!(prop.heads[0].contains(&11));
        assert!(prop.heads[0].contains(&12));
        let mut seen = Vec::new();
        for h in &prop.heads {
            for &t in h {
                if !seen.contains(&t) {
                    seen.push(t);
                }
            }
        }
        assert_eq!(seen, vec![11, 12]);
    });
}

#[test]
fn engine_step_medusa_respects_max_draft_tokens_cap() {
    let config = SpecDecodeConfig {
        mode: DraftMode::Medusa,
        max_draft_tokens: 3,
        ..Default::default()
    };
    let max = config.max_draft_tokens;
    let heads: Vec<Box<dyn MedusaHead>> = vec![
        Box::new(MockMedusaHead::new(vec![1, 2, 3, 4, 5, 6, 7, 8])),
        Box::new(MockMedusaHead::new(vec![9, 10, 11, 12])),
    ];
    let prop = MedusaProposal::from_heads(
        &heads,
        &[1, 2, 3],
        &TreeTopology { width: 4, depth: 2 },
        max,
    );
    let total: usize = prop.heads.iter().map(|h| h.len()).sum();
    assert!(total <= 3, "got {} total tokens, exceeds cap", total);
}

#[test]
fn medusa_head_propose_returns_at_most_k_tokens() {
    let head = MockMedusaHead::new(vec![1, 2, 3, 4, 5]);
    let out = head.propose(&[9, 8, 7], 3);
    assert_eq!(out.len(), 3);
    assert_eq!(out, vec![1, 2, 3]);
}

#[test]
fn medusa_head_propose_handles_kv_shorter_than_ngram_window() {
    let head = MockMedusaHead::new(vec![42]);
    let out = head.propose(&[1], 5);
    assert!(out.len() <= 5);
    assert!(!out.is_empty());
}

// ---------------------------------------------------------------------------
// MedusaProposal::from_heads edge cases
// ---------------------------------------------------------------------------

#[test]
fn medusa_proposal_from_heads_with_empty_heads_returns_empty() {
    let heads: Vec<Box<dyn MedusaHead>> = vec![];
    let prop =
        MedusaProposal::from_heads(&heads, &[1, 2, 3], &TreeTopology { width: 4, depth: 2 }, 8);
    assert!(
        prop.heads.is_empty(),
        "empty heads input must yield empty proposal"
    );
    assert_eq!(prop.total(), 0);
    assert!(prop.flat_tokens().is_empty());
}

#[test]
fn medusa_proposal_from_heads_single_head_proposes_in_order() {
    let heads: Vec<Box<dyn MedusaHead>> = vec![Box::new(MockMedusaHead::new(vec![10, 20, 30]))];
    let prop = MedusaProposal::from_heads(&heads, &[], &TreeTopology { width: 4, depth: 3 }, 3);
    assert_eq!(prop.heads.len(), 1);
    assert_eq!(prop.heads[0], vec![10, 20, 30]);
}

#[test]
fn medusa_proposal_from_heads_deduplicates_across_heads_preserving_first_seen() {
    let heads: Vec<Box<dyn MedusaHead>> = vec![
        Box::new(MockMedusaHead::new(vec![5, 6])),
        Box::new(MockMedusaHead::new(vec![5, 7])),
    ];
    let prop = MedusaProposal::from_heads(&heads, &[], &TreeTopology { width: 4, depth: 2 }, 10);
    let flat = prop.flat_tokens();
    assert_eq!(flat, vec![5, 6, 7]);
}

#[test]
fn medusa_proposal_from_heads_empty_heads_returns_empty() {
    let heads: Vec<Box<dyn MedusaHead>> = vec![];
    let prop = MedusaProposal::from_heads(&heads, &[], &TreeTopology { width: 4, depth: 2 }, 16);
    assert!(prop.heads.is_empty());
    assert_eq!(prop.total(), 0);
    assert!(prop.flat_tokens().is_empty());
    assert_eq!(prop.tree, TreeTopology { width: 4, depth: 2 });
}

// ---------------------------------------------------------------------------
// dedup_preserve edge cases
// ---------------------------------------------------------------------------

#[test]
fn dedup_preserve_all_duplicates_returns_single_element() {
    let result = dedup_preserve(vec![7, 7, 7, 7, 7]);
    assert_eq!(
        result,
        vec![7],
        "all-duplicate input must collapse to one element"
    );
}

#[test]
fn dedup_preserve_no_duplicates_preserves_all() {
    let input = vec![10, 20, 30, 40, 50];
    let result = dedup_preserve(input.clone());
    assert_eq!(
        result, input,
        "all-unique input must be preserved unchanged"
    );
}

#[test]
fn dedup_preserve_mixed_returns_first_seen_order() {
    let result = dedup_preserve(vec![1, 2, 1, 3, 2, 4]);
    assert_eq!(result, vec![1, 2, 3, 4]);
}
