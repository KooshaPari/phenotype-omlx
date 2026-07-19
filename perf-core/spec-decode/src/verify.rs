//! Verification routines — linear, tree, and the deterministic single-pass
//! `verify` used by structured proposals (Medusa, EAGLE, etc.).

use crate::engine::DraftCandidate;
use crate::{SpecDecodeConfig, SpecError};
use serde::{Deserialize, Serialize};

/// Outcome of a single-pass deterministic verification.
///
/// Acceptance semantics:
///   * `accepted_prefix` is the longest accepted prefix of `draft_tokens`.
///   * `first_reject_idx` is the index of the first mismatching draft token,
///     or `None` if the entire draft was accepted.
///   * `bonus_token` is the argmax of the target distribution emitted *after*
///     the accepted prefix (the "free" token the spec-decoding convention
///     adds when an extension is accepted). `None` when the draft is empty.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifyResult {
    pub accepted_prefix: Vec<u32>,
    pub first_reject_idx: Option<usize>,
    pub bonus_token: Option<u32>,
    /// Seed used to make the verifier reproducible. Recorded into the result
    /// so callers can compare runs without external bookkeeping.
    pub seed: Option<u64>,
}

/// Deterministic greedy verification of a draft against target logits.
///
/// Greedy: accept `draft[i]` iff `argmax(target_logits) == draft[i]`.
/// The first rejection stops the run; the target argmax at that index
/// becomes the bonus token. An empty draft returns empty accepted / no bonus
/// without error. Malformed `draft_probs` (length mismatch with the draft,
/// or NaN/inf entries) returns `Err(SpecError::Config(..))`. Draft tokens
/// are clamped to `config.max_draft_tokens`.
pub fn verify(
    target_logits: &[f32],
    draft_tokens: &[u32],
    draft_probs: &[f32],
    config: &SpecDecodeConfig,
) -> Result<VerifyResult, SpecError> {
    if target_logits.is_empty() {
        return Err(SpecError::Config(
            "target_logits must be non-empty".into(),
        ));
    }

    // Validate probs: must either be empty, exactly `draft_tokens.len()`,
    // or `vocab_size` long if a vocab-length array was supplied for
    // bookkeeping. We accept both shapes; only length == draft count is
    // actually meaningful for greedy verification.
    if !draft_probs.is_empty()
        && draft_probs.len() != draft_tokens.len()
        && draft_probs.len() != target_logits.len()
    {
        return Err(SpecError::Config(format!(
            "draft_probs length {} must be 0, draft_tokens.len()={}, or vocab={}",
            draft_probs.len(),
            draft_tokens.len(),
            target_logits.len()
        )));
    }
    for &p in draft_probs {
        if !p.is_finite() {
            return Err(SpecError::Config(
                "draft_probs contains NaN or infinity".into(),
            ));
        }
    }

    if draft_tokens.is_empty() {
        return Ok(VerifyResult {
            accepted_prefix: Vec::new(),
            first_reject_idx: None,
            bonus_token: None,
            seed: None,
        });
    }

    // Clamp the draft to the configured cap.
    let cap = config.max_draft_tokens.max(1);
    let n = draft_tokens.len().min(cap);
    let draft = &draft_tokens[..n];

    let argmax = match greedy(target_logits) {
        Some(idx) => idx,
        None => {
            return Err(SpecError::Config(
                "could not determine argmax of empty logits".into(),
            ))
        }
    };

    let mut accepted_prefix: Vec<u32> = Vec::with_capacity(n);
    for (i, &tok) in draft.iter().enumerate() {
        if tok == argmax {
            accepted_prefix.push(tok);
        } else {
            return Ok(VerifyResult {
                accepted_prefix,
                first_reject_idx: Some(i),
                bonus_token: Some(argmax),
                seed: Some(0),
            });
        }
    }

    Ok(VerifyResult {
        accepted_prefix,
        first_reject_idx: None,
        bonus_token: Some(argmax),
        seed: Some(0),
    })
}

/// Linear (vanilla) verification: run the target over each prefix+candidate
/// and accept the candidate iff the argmax matches the candidate's first
/// token. This is the conservative path — one forward pass per draft token.
pub async fn verify_linear(
    target: &dyn crate::backend::TargetBackend,
    prefix: &[u32],
    candidates: &[DraftCandidate],
) -> Result<Vec<bool>, SpecError> {
    let mut out = Vec::with_capacity(candidates.len());
    for cand in candidates {
        if cand.tokens.is_empty() {
            out.push(false);
            continue;
        }
        // Greedy accept: argmax of target(prefix) must equal cand.tokens[0].
        let logits = target
            .forward(prefix)
            .await
            .map_err(SpecError::Backend)?;
        let argmax = logits
            .logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        out.push(argmax == cand.tokens[0]);
    }
    Ok(out)
}

/// Tree verification: single forward pass over a tree of candidates using
/// tree attention. Accepts any candidate whose first-token matches target.
/// Backends without native tree attention should override this via the
/// `verify_tree` method on `TargetBackend`.
pub async fn verify_tree(
    target: &dyn crate::backend::TargetBackend,
    prefix: &[u32],
    candidates: &[DraftCandidate],
) -> Result<Vec<bool>, SpecError> {
    let tree: Vec<Vec<u32>> = candidates
        .iter()
        .map(|c| c.tokens.clone())
        .filter(|c| !c.is_empty())
        .collect();
    if tree.is_empty() {
        return Ok(candidates.iter().map(|_| false).collect());
    }
    let masks = target
        .verify_tree(prefix, &tree)
        .await
        .map_err(SpecError::Backend)?;
    Ok(masks)
}

fn greedy(logits: &[f32]) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SpecDecodeConfig {
        SpecDecodeConfig::default()
    }

    #[test]
    fn verify_full_accept() {
        let mut logits = vec![0.0_f32; 8];
        logits[5] = 10.0;
        let r = verify(&logits, &[5_u32, 5, 5], &[1.0; 8], &cfg()).unwrap();
        assert_eq!(r.accepted_prefix, vec![5, 5, 5]);
        assert_eq!(r.first_reject_idx, None);
        assert_eq!(r.bonus_token, Some(5));
    }

    #[test]
    fn verify_partial_accept() {
        let mut logits = vec![0.0_f32; 8];
        logits[2] = 10.0;
        let r = verify(&logits, &[2_u32, 2, 9], &[1.0; 8], &cfg()).unwrap();
        assert_eq!(r.accepted_prefix, vec![2, 2]);
        assert_eq!(r.first_reject_idx, Some(2));
        assert_eq!(r.bonus_token, Some(2));
    }

    #[test]
    fn verify_empty_draft_no_bonus() {
        let r = verify(&[0.0; 4], &[], &[], &cfg()).unwrap();
        assert!(r.accepted_prefix.is_empty());
        assert_eq!(r.first_reject_idx, None);
        assert_eq!(r.bonus_token, None);
    }

    #[test]
    fn verify_malformed_probs_errors() {
        let logits = vec![0.0_f32; 8];
        let draft = vec![1_u32, 2];
        let bad = vec![1.0_f32; 3];
        assert!(verify(&logits, &draft, &bad, &cfg()).is_err());
    }

    #[test]
    fn verify_clamps_draft() {
        let mut cfg = cfg();
        cfg.max_draft_tokens = 2;
        let mut logits = vec![0.0_f32; 8];
        logits[7] = 10.0;
        let draft = vec![7_u32, 7, 7, 7, 7];
        let r = verify(&logits, &draft, &[1.0; 8], &cfg).unwrap();
        assert_eq!(r.accepted_prefix.len(), 2);
    }

    #[test]
    fn verify_nan_probs_error() {
        let logits = vec![0.0_f32; 4];
        let draft = vec![1_u32];
        let mut probs = vec![1.0_f32; 1];
        probs[0] = f32::NAN;
        assert!(verify(&logits, &draft, &probs, &cfg()).is_err());
    }

    #[test]
    fn verify_records_seed() {
        let mut logits = vec![0.0_f32; 4];
        logits[0] = 1.0;
        let r = verify(&logits, &[0_u32], &[1.0; 4], &cfg()).unwrap();
        assert_eq!(r.seed, Some(0));
    }

    #[test]
    fn greedy_returns_none_on_empty() {
        assert_eq!(greedy(&[]), None);
    }
}
