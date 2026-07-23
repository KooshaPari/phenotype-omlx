//! The speculative decoding engine itself — orchestrates draft + verify.
//!
//! The engine wraps draft proposals and verification, accumulating per-layer
//! state in an [`EngineState`](crate::EngineState) that callers can observe
//! and reset through dedicated accessors.

use crate::backend::{DraftBackend, TargetBackend};
use crate::proposal::{MedusaHead, MedusaProposal, TreeTopology};
use crate::state::EngineState;
use crate::tree_proposal;
use crate::verify::{verify as verify_draft, verify_linear, verify_tree, VerifyResult};
use crate::{AcceptedToken, DraftMode, ProposalMode, SpecDecodeConfig, SpecError, StepResult};

/// Cancellation token: returns `true` when the caller wants to abort.
pub type CancelFn = dyn Fn() -> bool;

/// Statistics counters exposed to Python / FFI.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct SpecStats {
    pub drafted: usize,
    pub accepted: usize,
    pub steps: usize,
    pub rejection_rate: f32,
}

impl SpecStats {
    pub fn acceptance_rate(&self) -> f32 {
        if self.drafted == 0 {
            1.0
        } else {
            self.accepted as f32 / self.drafted as f32
        }
    }
}

/// A draft candidate with attached metadata for verification.
#[derive(Debug, Clone)]
pub struct DraftCandidate {
    pub tokens: Vec<u32>,
    pub tree_path: Option<Vec<usize>>,
}

/// The speculative decoding engine.
pub struct SpecDecodeEngine {
    pub config: SpecDecodeConfig,
    pub target: Box<dyn TargetBackend>,
    pub draft: Option<Box<dyn DraftBackend>>,
    pub stats: SpecStats,
    /// Per-call state: most recent KV-cache content for SameModel drafting.
    pub seen_tokens: Vec<u32>,
    /// Proposal strategy — Medusa or EAGLE-3 tree-based.
    pub proposal_mode: ProposalMode,
    /// Observable engine state — KV length, drafted/accepted counters, history.
    state: EngineState,
}

impl SpecDecodeEngine {
    pub fn new(
        config: SpecDecodeConfig,
        target: Box<dyn TargetBackend>,
        draft: Option<Box<dyn DraftBackend>>,
    ) -> Self {
        Self {
            config,
            target,
            draft,
            stats: SpecStats::default(),
            seen_tokens: Vec::new(),
            proposal_mode: ProposalMode::default(),
            state: EngineState::new(),
        }
    }

    pub fn with_proposal_mode(mut self, mode: ProposalMode) -> Self {
        self.proposal_mode = mode;
        self
    }

    /// Snapshot of the current engine state — pure data, no locks required.
    pub fn state(&self) -> EngineState {
        self.state.snapshot()
    }

    /// Reset every counter and history. Does not touch `config`, `target`,
    /// or `draft`.
    pub fn reset_state(&mut self) {
        self.state.reset();
        self.stats = SpecStats::default();
        self.seen_tokens.clear();
    }

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

    /// Run one draft-then-verify step.
    pub async fn step(&mut self, prefix: &[u32]) -> Result<StepResult, SpecError> {
        self.step_cancellable(prefix, &[], || false).await
    }

    /// Same as [`step`] but cooperative-cancellation-aware.
    ///
    /// `cancel` is polled once at the entry and once between draft and verify.
    /// When it returns `true` the engine returns an empty `StepResult`
    /// without mutating any state — counters, history, and KV position are
    /// left untouched.
    pub async fn step_cancellable(
        &mut self,
        prefix: &[u32],
        heads: &[Box<dyn MedusaHead>],
        cancel: impl Fn() -> bool,
    ) -> Result<StepResult, SpecError> {
        if cancel() {
            return Ok(StepResult {
                accepted: Vec::new(),
                drafted: 0,
                finished: false,
            });
        }

        // 1. Draft
        let candidates = self.propose(prefix, heads).await?;

        if cancel() {
            return Ok(StepResult {
                accepted: Vec::new(),
                drafted: 0,
                finished: false,
            });
        }

        // 2. Verify
        let accepted = match self.config.mode {
            DraftMode::Medusa => verify_tree(&*self.target, prefix, &candidates).await?,
            _ => verify_linear(&*self.target, prefix, &candidates).await?,
        };

        // 3. Build result
        let mut accepted_tokens: Vec<AcceptedToken> = Vec::with_capacity(accepted.len());
        let drafted = candidates.first().map(|c| c.tokens.len()).unwrap_or(0);

        for (i, &ok) in accepted.iter().enumerate() {
            if let Some(c) = candidates.get(i) {
                if ok {
                    if let Some(&tok) = c.tokens.first() {
                        accepted_tokens.push(AcceptedToken {
                            token_id: tok,
                            was_drafted: true,
                        });
                    }
                }
            }
        }

        // If everything was rejected, fall back to one greedy token.
        if accepted_tokens.is_empty() && self.config.fallback_on_reject {
            let out = self
                .target
                .forward(prefix)
                .await
                .map_err(SpecError::Backend)?;
            let next = greedy(&out.logits);
            accepted_tokens.push(AcceptedToken {
                token_id: next,
                was_drafted: false,
            });
            self.state.extend_kv(1);
            self.state.push_accepted(next);
            self.state.record_step(drafted, 1);
            return Ok(StepResult {
                accepted: accepted_tokens,
                drafted,
                finished: out.finished,
            });
        }

        let finished = candidates
            .first()
            .and_then(|c| {
                if c.tokens.is_empty() {
                    Some(true)
                } else {
                    None
                }
            })
            .unwrap_or(false);

        for t in &accepted_tokens {
            self.state.push_accepted(t.token_id);
            self.state.extend_kv(1);
        }
        self.state.record_step(drafted, accepted_tokens.len());

        Ok(StepResult {
            accepted: accepted_tokens,
            drafted,
            finished,
        })
    }

    /// Run a Medusa-mode step using the supplied draft heads.
    ///
    /// Builds a [`MedusaProposal`] from `heads`, then defers verification to
    /// the configured target backend's tree-attention path.
    pub async fn step_medusa(
        &mut self,
        prompt: &[u32],
        heads: &[Box<dyn MedusaHead>],
    ) -> Result<StepResult, SpecError> {
        let topology = TreeTopology {
            width: self.config.tree_width.max(1),
            depth: self.config.tree_depth.max(1),
        };
        let proposal =
            MedusaProposal::from_heads(heads, prompt, &topology, self.config.max_draft_tokens);

        if proposal.total() == 0 {
            return Ok(StepResult {
                accepted: Vec::new(),
                drafted: 0,
                finished: false,
            });
        }

        // Convert the proposal into a single DraftCandidate spanning all heads.
        let tokens = proposal.flat_tokens();
        let candidates = vec![DraftCandidate {
            tokens,
            tree_path: Some((0..proposal.heads.len()).collect()),
        }];
        let drafted = candidates.first().map(|c| c.tokens.len()).unwrap_or(0);

        // Verify via tree-attention path.
        let accepted = verify_tree(&*self.target, prompt, &candidates).await?;

        let mut accepted_tokens: Vec<AcceptedToken> = Vec::new();
        for (i, &ok) in accepted.iter().enumerate() {
            if ok {
                if let Some(c) = candidates.get(i) {
                    if let Some(&tok) = c.tokens.first() {
                        accepted_tokens.push(AcceptedToken {
                            token_id: tok,
                            was_drafted: true,
                        });
                    }
                }
            }
        }

        for t in &accepted_tokens {
            self.state.push_accepted(t.token_id);
            self.state.extend_kv(1);
        }
        self.state.record_step(drafted, accepted_tokens.len());
        self.seen_tokens.extend(prompt);
        for t in &accepted_tokens {
            self.seen_tokens.push(t.token_id);
        }

        Ok(StepResult {
            accepted: accepted_tokens,
            drafted,
            finished: false,
        })
    }

    /// Run an EAGLE-3 / P-EAGLE step using tree-based draft proposals.
    ///
    /// `branch_logits` are the per-depth candidate lists from the EAGLE-3
    /// draft head. The engine builds parallel trees, flattens every
    /// root-to-leaf path into a `DraftCandidate`, and verifies via tree
    /// attention.
    pub async fn step_eagle3(
        &mut self,
        prefix: &[u32],
        branch_logits: Vec<Vec<(u32, f32)>>,
    ) -> Result<StepResult, SpecError> {
        let eagle_config = match &self.proposal_mode {
            ProposalMode::Eagle3(cfg) => cfg.clone(),
            _ => tree_proposal::ParallelTreeConfig::default(),
        };

        let root_token = self.state.history.back().copied().unwrap_or(0);
        let trees =
            tree_proposal::create_parallel_trees(root_token, branch_logits, &eagle_config);

        let mut candidates: Vec<DraftCandidate> = Vec::new();
        for tree in &trees {
            for path in tree.leaf_paths() {
                if !path.is_empty() {
                    candidates.push(DraftCandidate {
                        tokens: path,
                        tree_path: None,
                    });
                }
            }
        }

        if candidates.is_empty() {
            return Ok(StepResult {
                accepted: Vec::new(),
                drafted: 0,
                finished: false,
            });
        }

        let drafted = candidates.first().map(|c| c.tokens.len()).unwrap_or(0);

        let accepted = verify_tree(&*self.target, prefix, &candidates).await?;

        let mut accepted_tokens: Vec<AcceptedToken> = Vec::new();
        for (i, &ok) in accepted.iter().enumerate() {
            if ok {
                if let Some(c) = candidates.get(i) {
                    if let Some(&tok) = c.tokens.first() {
                        accepted_tokens.push(AcceptedToken {
                            token_id: tok,
                            was_drafted: true,
                        });
                    }
                }
            }
        }

        for t in &accepted_tokens {
            self.state.push_accepted(t.token_id);
            self.state.extend_kv(1);
        }
        self.state.record_step(drafted, accepted_tokens.len());
        self.seen_tokens.extend(prefix);
        for t in &accepted_tokens {
            self.seen_tokens.push(t.token_id);
        }

        Ok(StepResult {
            accepted: accepted_tokens,
            drafted,
            finished: false,
        })
    }

    /// Run a deterministic verification pass against the target's logits
    /// without performing any draft step — exposed so external callers
    /// (Python, FFI) can plug in custom draft proposals.
    pub fn verify_only(
        &self,
        target_logits: &[f32],
        draft_tokens: &[u32],
        draft_probs: &[f32],
    ) -> Result<VerifyResult, SpecError> {
        verify_draft(target_logits, draft_tokens, draft_probs, &self.config)
    }

    /// Produce draft candidates appropriate for the configured mode.
    async fn propose(
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

fn greedy(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Prompt-lookup / n-gram match: scan `seen` for the longest suffix of
/// `prefix` and emit the next `k` tokens from any matching continuation.
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
    'outer: for i in 0..=hay.len() - needle.len() {
        for j in 0..needle.len() {
            if hay[i + j] != needle[j] {
                continue 'outer;
            }
        }
        return Some(i);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendInfo, TargetOutput};
    use async_trait::async_trait;

    struct ConstantTarget(u32);
    #[async_trait]
    impl TargetBackend for ConstantTarget {
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
            // accept candidates whose first token == our constant
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

    #[test]
    fn state_initially_zero() {
        let e = SpecDecodeEngine::new(
            SpecDecodeConfig::default(),
            Box::new(ConstantTarget(1)),
            None,
        );
        let s = e.state();
        assert_eq!(s.kv_len, 0);
        assert_eq!(s.drafted_total, 0);
    }

    #[test]
    fn reset_state_zeroes_everything() {
        let mut e = SpecDecodeEngine::new(
            SpecDecodeConfig::default(),
            Box::new(ConstantTarget(1)),
            None,
        );
        e.state.extend_kv(10);
        e.state.record_step(4, 3);
        e.seen_tokens.extend_from_slice(&[1, 2, 3]);
        e.reset_state();
        let s = e.state();
        assert_eq!(s.kv_len, 0);
        assert_eq!(s.drafted_total, 0);
        assert!(s.history.is_empty());
        assert!(e.seen_tokens.is_empty());
    }

    #[tokio::test]
    async fn step_cancellable_no_cancel_returns_step() {
        let mut e = SpecDecodeEngine::new(
            SpecDecodeConfig::default(),
            Box::new(ConstantTarget(2)),
            None,
        );
        let r = e.step_cancellable(&[1, 2, 3], &[], || false).await.unwrap();
        // accept at least 0 tokens; do not panic.
        let _ = r;
    }

    #[tokio::test]
    async fn step_cancellable_with_cancel_returns_empty() {
        let mut e = SpecDecodeEngine::new(
            SpecDecodeConfig::default(),
            Box::new(ConstantTarget(2)),
            None,
        );
        let before = e.state();
        let r = e.step_cancellable(&[1, 2, 3], &[], || true).await.unwrap();
        assert!(r.accepted.is_empty());
        assert_eq!(r.drafted, 0);
        assert!(!r.finished);
        assert_eq!(e.state(), before);
    }

    #[test]
    fn verify_only_is_deterministic() {
        let e = SpecDecodeEngine::new(
            SpecDecodeConfig::default(),
            Box::new(ConstantTarget(5)),
            None,
        );
        let mut logits = vec![0.0_f32; 16];
        logits[5] = 10.0;
        let r = e.verify_only(&logits, &[5_u32, 5, 5], &[1.0; 16]).unwrap();
        assert_eq!(r.accepted_prefix.len(), 3);
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

    // -- EAGLE-3 / ProposalMode tests --------------------------------------------------

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
        // ConstantTarget(5) accepts any candidate whose first token == 5.
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
}
