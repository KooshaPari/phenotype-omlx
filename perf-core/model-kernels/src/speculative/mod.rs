//! Speculative decoding proposal kernels.
//!
//! This module is the canonical stub for the "multi-token-prediction"
//! (MTP) heads used by DeepSeek-V3 / EAGLE-style speculative decoders.
//! It does *not* own tree masking or Medusa-style head dispatch (those
//! live in `spec-decode`); it provides a minimal structural kernel that
//! the spec-decode engine can compose with. The Medusa engine in
//! `perf-core/spec-decode` is intentionally untouched by this crate.
//!
//! # MTP depth = 1
//!
//! For DeepSeek-V3's MTP-1, the proposal is exactly one candidate
//! token per draft offset. [`MtpProposal`] therefore carries equal-
//! length `tokens` and `accepted_mask` vectors of length
//! `draft_offsets.len()`.

use crate::error::{KernelError, Result};

/// A multi-token-prediction proposal emitted by a single MTP head.
///
/// `tokens[i]` is the candidate token at draft offset
/// `draft_offsets[i]`. `accepted_mask[i]` is `true` iff the verifier
/// decided to commit to that candidate (see [`mtp_verify`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpProposal {
    /// Candidate tokens, one per draft offset, in offset order.
    pub tokens: Vec<u32>,
    /// Per-position acceptance mask, set by the verifier.
    pub accepted_mask: Vec<bool>,
}

impl MtpProposal {
    /// Number of draft positions in this proposal.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// `true` iff the proposal has no draft positions.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Produce an MTP proposal from a single logits tensor.
///
/// `seed_logits` is laid out as `[draft_offsets.len(), vocab]`: for
/// each draft offset `i`, the proposal token is
/// `argmax(seed_logits[offsets[i] * vocab .. offsets[i] * vocab + vocab])`.
///
/// `draft_offsets` selects the head row to read for each draft step.
/// All offsets must fit inside `seed_logits` when multiplied by `vocab`.
/// The output mask is initialised to all-`false`; the verifier is
/// expected to fill it via [`mtp_verify`].
pub fn mtp_propose(
    seed_logits: &[f32],
    draft_offsets: &[usize],
    vocab: usize,
) -> Result<MtpProposal> {
    if vocab == 0 {
        return Err(KernelError::ZeroDimension {
            what: "vocab",
            got: 0,
        });
    }
    if seed_logits.len() % vocab != 0 {
        return Err(KernelError::BadBufferLength {
            what: "seed_logits",
            expected: (seed_logits.len() / vocab + 1) * vocab,
            got: seed_logits.len(),
        });
    }
    let rows = seed_logits.len() / vocab;
    let mut tokens = Vec::with_capacity(draft_offsets.len());
    for &off in draft_offsets {
        if off >= rows {
            return Err(KernelError::ExpertOutOfRange {
                num_experts: rows,
                got: off,
            });
        }
        let row = &seed_logits[off * vocab..off * vocab + vocab];
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        tokens.push(best_idx as u32);
    }
    Ok(MtpProposal {
        tokens,
        accepted_mask: vec![false; draft_offsets.len()],
    })
}

/// Verify a proposal against a verifier logits tensor.
///
/// `verifier_logits` has the same `[n_drafts, vocab]` layout as
/// `seed_logits`. For each draft position `i`, the verifier computes
/// the softmax of `verifier_logits[i, :]` and accepts the proposal
/// token iff the verifier's softmax probability for the proposal
/// token is at least `accepted_threshold`.
///
/// Returns a `Vec<bool>` mask aligned with `proposal.tokens`. The
/// proposal is left unchanged.
pub fn mtp_verify(
    proposal: &MtpProposal,
    verifier_logits: &[f32],
    vocab: usize,
    accepted_threshold: f32,
) -> Result<Vec<bool>> {
    if vocab == 0 {
        return Err(KernelError::ZeroDimension {
            what: "vocab",
            got: 0,
        });
    }
    if !(0.0..=1.0).contains(&accepted_threshold) {
        return Err(KernelError::OutOfRange {
            what: "accepted_threshold",
            min: 0.0,
            max: 1.0,
            got: accepted_threshold,
        });
    }
    if verifier_logits.len() != proposal.len() * vocab {
        return Err(KernelError::BadBufferLength {
            what: "verifier_logits",
            expected: proposal.len() * vocab,
            got: verifier_logits.len(),
        });
    }
    let mut mask = Vec::with_capacity(proposal.len());
    for (i, &tok) in proposal.tokens.iter().enumerate() {
        let row = &verifier_logits[i * vocab..i * vocab + vocab];
        if (tok as usize) >= vocab {
            return Err(KernelError::ExpertOutOfRange {
                num_experts: vocab,
                got: tok as usize,
            });
        }
        // Numerically-stable softmax.
        let mut max = f32::NEG_INFINITY;
        for &v in row {
            if v > max {
                max = v;
            }
        }
        let mut sum = 0.0f32;
        let mut probs = vec![0.0f32; vocab];
        for (j, &v) in row.iter().enumerate() {
            let e = (v - max).exp();
            probs[j] = e;
            sum += e;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for p in probs.iter_mut() {
                *p *= inv;
            }
        }
        mask.push(probs[tok as usize] >= accepted_threshold);
    }
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propose_rejects_zero_vocab() {
        let logits = vec![0.0f32; 4];
        let err = mtp_propose(&logits, &[0, 1], 0).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn propose_rejects_misaligned_logits() {
        // 5 elements with vocab=4 -> not a multiple.
        let logits = vec![0.0f32; 5];
        let err = mtp_propose(&logits, &[0], 4).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn propose_empty_offsets_yields_empty_proposal() {
        let logits = vec![0.0f32; 12];
        let p = mtp_propose(&logits, &[], 4).unwrap();
        assert!(p.is_empty());
        assert_eq!(p.tokens, Vec::<u32>::new());
        assert_eq!(p.accepted_mask, Vec::<bool>::new());
    }

    #[test]
    fn propose_argmax_is_per_row() {
        // Two rows, vocab=3.
        // Row 0: max at index 2 (logit 7.0).
        // Row 1: max at index 0 (logit 9.0).
        let logits = vec![1.0f32, 2.0, 7.0, 9.0, 3.0, 4.0];
        let p = mtp_propose(&logits, &[0, 1], 3).unwrap();
        assert_eq!(p.tokens, vec![2, 0]);
        assert_eq!(p.accepted_mask, vec![false, false]);
    }

    #[test]
    fn propose_rejects_out_of_range_offset() {
        let logits = vec![0.0f32; 6];
        let err = mtp_propose(&logits, &[0, 5], 3).unwrap_err();
        assert!(matches!(err, KernelError::ExpertOutOfRange { .. }));
    }

    #[test]
    fn verify_accepts_high_confidence_verifier() {
        // Seed proposal: token 0.
        let proposal = MtpProposal {
            tokens: vec![0],
            accepted_mask: vec![false],
        };
        // Verifier: logits that put ~100% mass on token 0.
        let verifier = vec![100.0f32, 0.0, 0.0];
        let mask = mtp_verify(&proposal, &verifier, 3, 0.5).unwrap();
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn verify_rejects_low_confidence_verifier() {
        // Proposal token 0; verifier logits are uniform -> prob = 1/vocab
        // = 0.5 with vocab=2. Threshold 0.9 -> rejected.
        let proposal = MtpProposal {
            tokens: vec![0],
            accepted_mask: vec![false],
        };
        let verifier = vec![0.0f32, 0.0];
        let mask = mtp_verify(&proposal, &verifier, 2, 0.9).unwrap();
        assert_eq!(mask, vec![false]);
    }

    #[test]
    fn verify_threshold_zero_accepts_everything() {
        let proposal = MtpProposal {
            tokens: vec![0, 2],
            accepted_mask: vec![false, false],
        };
        let verifier = vec![-5.0f32, -5.0, -5.0, -5.0, -5.0, -5.0];
        let mask = mtp_verify(&proposal, &verifier, 3, 0.0).unwrap();
        assert_eq!(mask, vec![true, true]);
    }

    #[test]
    fn verify_rejects_mismatched_vocab_length() {
        let proposal = MtpProposal {
            tokens: vec![0, 1],
            accepted_mask: vec![false, false],
        };
        // We claim vocab=4 but only provide 6 floats.
        let verifier = vec![0.0f32; 6];
        let err = mtp_verify(&proposal, &verifier, 4, 0.5).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn verify_rejects_threshold_out_of_range() {
        let proposal = MtpProposal {
            tokens: vec![0],
            accepted_mask: vec![false],
        };
        let verifier = vec![0.0f32, 0.0];
        let err = mtp_verify(&proposal, &verifier, 2, 1.5).unwrap_err();
        assert!(matches!(err, KernelError::OutOfRange { .. }));
    }

    #[test]
    fn verify_rejects_proposal_token_outside_vocab() {
        let proposal = MtpProposal {
            tokens: vec![7], // out of vocab=3
            accepted_mask: vec![false],
        };
        let verifier = vec![0.0f32; 3];
        let err = mtp_verify(&proposal, &verifier, 3, 0.5).unwrap_err();
        assert!(matches!(err, KernelError::ExpertOutOfRange { .. }));
    }
}
