//! Proposal generation — draft candidates via SameModel, DraftModel, or EAGLE-3.

use super::{DraftCandidate, SpecDecodeEngine};
use crate::proposal::MedusaHead;
use crate::tree_proposal;
use crate::{DraftMode, SpecError};

impl SpecDecodeEngine {
    /// Generate draft proposals using EAGLE-3 tree-based approach.
    pub fn propose_eagle3(
        &self,
        branch_logits: Vec<Vec<(u32, f32)>>,
        config: &tree_proposal::ParallelTreeConfig,
    ) -> tree_proposal::DraftTree {
        let root_token = self.state.history.back().copied().unwrap_or(0);
        let trees =
            tree_proposal::create_parallel_trees(root_token, branch_logits, config);
        trees
            .into_iter()
            .next()
            .unwrap_or_else(|| tree_proposal::DraftTree {
                root: tree_proposal::DraftNode {
                    token_id: root_token,
                    probability: 1.0,
                    children: vec![],
                },
                depth: 0,
                total_leaves: 1,
            })
    }

    /// Submit a single prompt and generate up to `max_new_tokens` accepted tokens.
    pub async fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, SpecError> {
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let mut history = prompt.to_vec();
        self.seen_tokens.extend_from_slice(prompt);
        self.state.extend_kv(prompt.len());

        while generated.len() < max_new_tokens {
            let step = self.step(&history).await?;
            let n = step.accepted.len();
            for t in &step.accepted {
                generated.push(t.token_id);
                history.push(t.token_id);
                self.seen_tokens.push(t.token_id);
                self.state.push_accepted(t.token_id);
                self.state.extend_kv(1);
            }
            self.stats.drafted += step.drafted;
            self.stats.accepted += n;
            self.stats.steps += 1;
            self.state.record_step(step.drafted, n);
            if step.finished || n == 0 {
                break;
            }
        }

        self.stats.rejection_rate = 1.0 - self.stats.acceptance_rate();
        Ok(generated)
    }

    /// Produce draft candidates appropriate for the configured mode.
    pub(crate) async fn propose(
        &mut self,
        prefix: &[u32],
        _heads: &[Box<dyn MedusaHead>],
    ) -> Result<Vec<DraftCandidate>, SpecError> {
        match self.config.mode {
            DraftMode::SameModel => {
                Ok(
                    prompt_lookup(prefix, &self.seen_tokens, self.config.max_draft_tokens)
                        .into_iter()
                        .map(|tokens| DraftCandidate {
                            tokens,
                            tree_path: None,
                        })
                        .collect(),
                )
            }
            DraftMode::DraftModel => {
                let draft = self.draft.as_ref().ok_or(SpecError::DraftNotLoaded)?;
                let tokens = draft
                    .draft(prefix, self.config.max_draft_tokens)
                    .await
                    .map_err(SpecError::Backend)?;
                Ok(vec![DraftCandidate {
                    tokens,
                    tree_path: None,
                }])
            }
            DraftMode::Medusa => {
                // Medusa is driven through `step_medusa`; `propose` here is
                // only called from the linear / SameModel paths and must
                // not synthesize candidates it didn't observe.
                Ok(Vec::new())
            }
        }
    }
}

fn prompt_lookup(prefix: &[u32], seen: &[u32], k: usize) -> Vec<Vec<u32>> {
    if prefix.is_empty() || seen.len() <= prefix.len() {
        return vec![Vec::new()];
    }
    // Walk back the longest suffix that appears earlier in `seen`.
    for n in (1..=prefix.len().min(64)).rev() {
        let needle = &prefix[prefix.len() - n..];
        if let Some(pos) = find_subseq(seen[..seen.len() - prefix.len()].as_ref(), needle) {
            let start = pos + n;
            let end = (start + k).min(seen.len());
            if start < seen.len() && start < end {
                return vec![seen[start..end].to_vec()];
            }
        }
    }
    vec![Vec::new()]
}

fn find_subseq(hay: &[u32], needle: &[u32]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    let mut match_len = 0usize;
    let mut start = 0usize;
    for (i, &h) in hay.iter().enumerate() {
        if h == needle[match_len] {
            if match_len == 0 {
                start = i;
            }
            match_len += 1;
            if match_len == needle.len() {
                return Some(start);
            }
        } else {
            match_len = 0;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendInfo, TargetOutput};
    use crate::{ProposalMode, SpecDecodeConfig, SpecDecodeEngine};
    use async_trait::async_trait;

    struct ConstantTarget(u32);
    #[async_trait]
    impl crate::backend::TargetBackend for ConstantTarget {
        async fn forward(&self, _: &[u32]) -> Result<TargetOutput, String> {
            let mut logits = vec![0.0_f32; 64];
            logits[self.0 as usize] = 10.0;
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
            Ok(candidates
                .iter()
                .map(|c| c.first().copied() == Some(self.0))
                .collect())
        }
        fn info(&self) -> BackendInfo {
            BackendInfo {
                engine: "test".into(),
                model_id: "constant".into(),
                device: "cpu".into(),
                dtype: "f32".into(),
                kv_cache_type: None,
            }
        }
    }

    struct AcceptAllTarget;
    #[async_trait]
    impl crate::backend::TargetBackend for AcceptAllTarget {
        async fn forward(&self, _: &[u32]) -> Result<TargetOutput, String> {
            Ok(TargetOutput {
                logits: vec![0.0; 64],
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

    #[test]
    fn propose_eagle3_with_known_logits_produces_valid_tree() {
        let e = SpecDecodeEngine::new(
            SpecDecodeConfig::default(),
            Box::new(ConstantTarget(1)),
            None,
        );
        let logits = vec![
            vec![(10, 0.6), (20, 0.3), (30, 0.1)],
            vec![(100, 0.5), (110, 0.4)],
        ];
        let config = tree_proposal::ParallelTreeConfig {
            num_parallel_branches: 2,
            max_depth: 2,
            max_branches_per_node: 2,
            probability_threshold: 0.01,
        };
        let tree = e.propose_eagle3(logits, &config);
        assert!(tree.depth >= 1);
        assert!(tree.total_leaves >= 1);
        assert_eq!(tree.root.token_id, 0);
    }

    #[test]
    fn propose_eagle3_empty_logits_returns_single_node() {
        let e = SpecDecodeEngine::new(
            SpecDecodeConfig::default(),
            Box::new(ConstantTarget(1)),
            None,
        );
        let config = tree_proposal::ParallelTreeConfig::default();
        let tree = e.propose_eagle3(vec![], &config);
        assert_eq!(tree.depth, 0);
        assert_eq!(tree.total_leaves, 1);
        assert_eq!(tree.root.token_id, 0);
    }

    #[test]
    fn propose_eagle3_with_history_uses_last_token_as_root() {
        let mut e = SpecDecodeEngine::new(
            SpecDecodeConfig::default(),
            Box::new(ConstantTarget(1)),
            None,
        );
        e.state.push_accepted(42);
        let logits = vec![vec![(10, 0.9)]];
        let config = tree_proposal::ParallelTreeConfig {
            num_parallel_branches: 1,
            max_depth: 1,
            max_branches_per_node: 1,
            probability_threshold: 0.01,
        };
        let tree = e.propose_eagle3(logits, &config);
        assert_eq!(tree.root.token_id, 42);
        assert_eq!(tree.root.children.len(), 1);
        assert_eq!(tree.root.children[0].token_id, 10);
    }

    #[test]
    fn proposal_mode_medusa_is_default() {
        let mode = ProposalMode::default();
        assert!(matches!(mode, ProposalMode::Medusa));
    }

    #[test]
    fn proposal_mode_eagle3_custom_config() {
        let config = tree_proposal::ParallelTreeConfig {
            num_parallel_branches: 8,
            max_depth: 16,
            max_branches_per_node: 4,
            probability_threshold: 0.001,
        };
        let mode = ProposalMode::Eagle3(config.clone());
        match mode {
            ProposalMode::Eagle3(cfg) => {
                assert_eq!(cfg.num_parallel_branches, 8);
                assert_eq!(cfg.max_depth, 16);
                assert_eq!(cfg.max_branches_per_node, 4);
                assert_eq!(cfg.probability_threshold, 0.001);
            }
            _ => panic!("expected Eagle3 variant"),
        }
    }

    #[test]
    fn engine_with_proposal_mode_eagle3() {
        let e = SpecDecodeEngine::new(
            SpecDecodeConfig::default(),
            Box::new(ConstantTarget(1)),
            None,
        )
        .with_proposal_mode(ProposalMode::Eagle3(
            tree_proposal::ParallelTreeConfig::default(),
        ));
        match &e.proposal_mode {
            ProposalMode::Eagle3(_) => {}
            _ => panic!("expected Eagle3"),
        }
    }

    #[tokio::test]
    async fn step_eagle3_with_matching_target_accepts_tokens() {
        let mut e = SpecDecodeEngine::new(
            SpecDecodeConfig::default(),
            Box::new(ConstantTarget(5)),
            None,
        )
        .with_proposal_mode(ProposalMode::Eagle3(
            tree_proposal::ParallelTreeConfig {
                num_parallel_branches: 1,
                max_depth: 2,
                max_branches_per_node: 2,
                probability_threshold: 0.01,
            },
        ));
        let logits = vec![
            vec![(5, 0.9), (3, 0.1)],
            vec![(5, 0.8), (7, 0.2)],
        ];
        let result = e.step_eagle3(&[1, 2, 3], logits).await.unwrap();
        assert!(!result.accepted.is_empty());
        assert_eq!(result.accepted[0].token_id, 5);
        assert!(result.accepted[0].was_drafted);
    }

    #[tokio::test]
    async fn step_eagle3_empty_logits_returns_empty() {
        let mut e = SpecDecodeEngine::new(
            SpecDecodeConfig::default(),
            Box::new(ConstantTarget(5)),
            None,
        );
        let result = e.step_eagle3(&[1, 2], vec![]).await.unwrap();
        assert!(result.accepted.is_empty());
        assert_eq!(result.drafted, 0);
    }

    #[test]
    fn engine_with_echokv_evicts_on_step() {
        use tree_proposal::ParallelTreeConfig;
        let config = ParallelTreeConfig::default();
        let engine = SpecDecodeEngine::new(
            SpecDecodeConfig::default(),
            Box::new(AcceptAllTarget),
            None,
        )
        .with_proposal_mode(ProposalMode::Eagle3(config))
        .with_echokv(16);

        assert!(engine.kv_cache.is_some());
        assert_eq!(engine.kv_cache.as_ref().unwrap().len(), 0);
    }

    // -- find_subseq edge cases -------------------------------------------------------

    #[test]
    fn find_subseq_at_start_of_haystack() {
        let hay = vec![1, 2, 3, 4, 5];
        let needle = vec![1, 2];
        assert_eq!(find_subseq(&hay, &needle), Some(0));
    }

    #[test]
    fn find_subseq_at_end_of_haystack() {
        let hay = vec![10, 20, 30, 40, 50];
        let needle = vec![40, 50];
        assert_eq!(find_subseq(&hay, &needle), Some(3));
    }

    #[test]
    fn find_subseq_in_middle_of_haystack() {
        let hay = vec![10, 20, 30, 40, 50];
        let needle = vec![20, 30, 40];
        assert_eq!(find_subseq(&hay, &needle), Some(1));
    }

    #[test]
    fn find_subseq_needle_not_found_returns_none() {
        let hay = vec![1, 2, 3];
        let needle = vec![4, 5];
        assert_eq!(find_subseq(&hay, &needle), None);
    }

    #[test]
    fn find_subseq_empty_needle_returns_none() {
        let hay = vec![1, 2, 3];
        assert_eq!(find_subseq(&hay, &[]), None);
    }

    #[test]
    fn find_subseq_hay_shorter_than_needle_returns_none() {
        let hay = vec![1, 2];
        let needle = vec![1, 2, 3];
        assert_eq!(find_subseq(&hay, &needle), None);
    }

    #[test]
    fn find_subseq_single_element_match() {
        let hay = vec![10, 20, 30];
        let needle = vec![20];
        assert_eq!(find_subseq(&hay, &needle), Some(1));
    }

    #[test]
    fn find_subseq_exact_match_returns_zero() {
        let hay = vec![7, 8, 9];
        let needle = vec![7, 8, 9];
        assert_eq!(find_subseq(&hay, &needle), Some(0));
    }

    #[test]
    fn find_subseq_needle_longer_than_hay_returns_none() {
        let hay = vec![1];
        let needle = vec![1, 2, 3, 4, 5];
        assert_eq!(find_subseq(&hay, &needle), None);
    }
}
