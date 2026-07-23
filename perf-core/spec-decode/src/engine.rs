//! The speculative decoding engine itself — orchestrates draft + verify.
//!
//! The engine wraps draft proposals and verification, accumulating per-layer
//! state in an [`EngineState`](crate::EngineState) that callers can observe
//! and reset through dedicated accessors.

use crate::backend::{DraftBackend, TargetBackend};
use crate::proposal::{MedusaHead, MedusaProposal, TreeTopology};
use crate::state::EngineState;
use crate::verify::{verify as verify_draft, verify_linear, verify_tree, VerifyResult};
use crate::{AcceptedToken, DraftMode, SpecDecodeConfig, SpecError, StepResult};

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
            state: EngineState::new(),
        }
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
            .and_then(|c| if c.tokens.is_empty() { Some(true) } else { None })
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
        let proposal = MedusaProposal::from_heads(
            heads,
            prompt,
            &topology,
            self.config.max_draft_tokens,
        );

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
            DraftMode::SameModel => Ok(prompt_lookup(prefix, &self.seen_tokens, self.config.max_draft_tokens)
                .into_iter()
                .map(|tokens| DraftCandidate { tokens, tree_path: None })
                .collect()),
            DraftMode::DraftModel => {
                let draft = self.draft.as_ref().ok_or(SpecError::DraftNotLoaded)?;
                let tokens = draft
                    .draft(prefix, self.config.max_draft_tokens)
                    .await
                    .map_err(SpecError::Backend)?;
                Ok(vec![DraftCandidate { tokens, tree_path: None }])
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
        let r = e
            .verify_only(&logits, &[5_u32, 5, 5], &[1.0; 16])
            .unwrap();
        assert_eq!(r.accepted_prefix.len(), 3);
    }
}
