//! The speculative decoding engine itself — orchestrates draft + verify.
//!
//! The engine wraps draft proposals and verification, accumulating per-layer
//! state in an [`EngineState`](crate::EngineState) that callers can observe
//! and reset through dedicated accessors.

mod propose;
mod verify;

use crate::backend::{DraftBackend, TargetBackend};
use crate::proposal::MedusaHead;
use crate::state::EngineState;
use crate::tree_proposal;
use crate::verify::{verify_tree, verify_linear};
use crate::{AcceptedToken, DraftMode, ProposalMode, SpecDecodeConfig, SpecError, StepResult};
use turbo_quant::echokv::{EchoKVCache, EchoKVConfig};

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
    pub(crate) state: EngineState,
    /// EchoKV cache for adaptive eviction during decode.
    kv_cache: Option<EchoKVCache>,
    /// Configuration for the EchoKV cache.
    kv_config: EchoKVConfig,
}

impl SpecDecodeEngine {
    pub fn new(
        config: SpecDecodeConfig,
        target: Box<dyn TargetBackend>,
        draft: Option<Box<dyn DraftBackend>>,
    ) -> Self {
        let kv_config = EchoKVConfig::default();
        Self {
            config,
            target,
            draft,
            stats: SpecStats::default(),
            seen_tokens: Vec::new(),
            proposal_mode: ProposalMode::default(),
            state: EngineState::new(),
            kv_cache: None,
            kv_config,
        }
    }

    pub fn with_proposal_mode(mut self, mode: ProposalMode) -> Self {
        self.proposal_mode = mode;
        self
    }

    pub fn with_echokv(mut self, max_cache_size: usize) -> Self {
        self.kv_config = EchoKVConfig {
            max_cache_size,
            ..self.kv_config
        };
        self.kv_cache = Some(EchoKVCache::new(self.kv_config.clone()));
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
    /// Builds a [`MedusaProposal`](crate::proposal::MedusaProposal) from
    /// `heads`, then defers verification to the configured target backend's
    /// tree-attention path.
    pub async fn step_medusa(
        &mut self,
        prompt: &[u32],
        heads: &[Box<dyn MedusaHead>],
    ) -> Result<StepResult, SpecError> {
        let topology = crate::proposal::TreeTopology {
            width: self.config.tree_width.max(1),
            depth: self.config.tree_depth.max(1),
        };
        let proposal = crate::proposal::MedusaProposal::from_heads(
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

        // Evict low-scoring KV entries before verification.
        if let Some(ref mut cache) = self.kv_cache {
            for &tok in &self.seen_tokens {
                cache.insert(tok as usize, 1.0);
            }
            let evicted = cache.evict();
            if !evicted.is_empty() {
                tracing::debug!(
                    evicted_count = evicted.len(),
                    remaining = cache.len(),
                    "echokv: evicted entries before verification"
                );
            }
        }

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
}

pub(crate) fn greedy(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
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
        let r = e
            .step_cancellable(&[1, 2, 3], &[], || false)
            .await
            .unwrap();
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
        let r = e
            .step_cancellable(&[1, 2, 3], &[], || true)
            .await
            .unwrap();
        assert!(r.accepted.is_empty());
        assert_eq!(r.drafted, 0);
        assert!(!r.finished);
        assert_eq!(e.state(), before);
    }
}
