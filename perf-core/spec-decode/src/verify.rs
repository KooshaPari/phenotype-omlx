//! Verification routines — linear and tree-attention.

use crate::backend::TargetBackend;
use crate::engine::DraftCandidate;

/// Linear (vanilla) verification: run the target over each prefix+candidate
/// and accept the candidate iff the argmax matches the candidate's first
/// token. This is the conservative path — one forward pass per draft token.
pub async fn verify_linear(
    target: &dyn TargetBackend,
    prefix: &[u32],
    candidates: &[DraftCandidate],
) -> Result<Vec<bool>, crate::SpecError> {
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
            .map_err(crate::SpecError::Backend)?;
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
    target: &dyn TargetBackend,
    prefix: &[u32],
    candidates: &[DraftCandidate],
) -> Result<Vec<bool>, crate::SpecError> {
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
        .map_err(crate::SpecError::Backend)?;
    Ok(masks)
}
