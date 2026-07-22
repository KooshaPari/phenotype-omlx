//! The speculative decoding engine itself — orchestrates draft + verify.

use crate::backend::{DraftBackend, TargetBackend};
use crate::verify::{verify_linear, verify_tree};
use crate::{AcceptedToken, DraftMode, SpecDecodeConfig, SpecError, StepResult};

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
        }
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

        while generated.len() < max_new_tokens {
            let step = self.step(&history).await?;
            let n = step.accepted.len();
            for t in &step.accepted {
                generated.push(t.token_id);
                history.push(t.token_id);
                self.seen_tokens.push(t.token_id);
            }
            self.stats.drafted += step.drafted;
            self.stats.accepted += n;
            self.stats.steps += 1;
            if step.finished || n == 0 {
                break;
            }
        }

        self.stats.rejection_rate = 1.0 - self.stats.acceptance_rate();
        Ok(generated)
    }

    /// Run one draft-then-verify step.
    pub async fn step(&mut self, prefix: &[u32]) -> Result<StepResult, SpecError> {
        // 1. Draft
        let candidates = self.propose(prefix).await?;

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

        Ok(StepResult {
            accepted: accepted_tokens,
            drafted,
            finished,
        })
    }

    /// Produce draft candidates appropriate for the configured mode.
    async fn propose(&mut self, prefix: &[u32]) -> Result<Vec<DraftCandidate>, SpecError> {
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
                // Tree expansion is performed inside verify_tree; here we just
                // emit an empty token set that the verifier will fill.
                Ok(vec![DraftCandidate {
                    tokens: Vec::new(),
                    tree_path: Some(Vec::new()),
                }])
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
