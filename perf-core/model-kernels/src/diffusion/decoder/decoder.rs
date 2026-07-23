//! `DiffusionDecoder`: multi-step orchestrator over
//! [`crate::diffusion::denoise::denoise_step`].
//!
//! See the parent module docs for the LLaDA / Dream acceptance trace
//! algorithm.

use crate::error::{KernelError, Result};

use super::super::denoise::{denoise_step, RemaskStrategy};
use super::report::DiffusionStepReport;

/// Multi-step diffusion decoder: orchestrates [`denoise_step`] calls
/// across a fixed number of steps with a configured remask strategy
/// and confidence threshold.
///
/// The decoder is pure: it does not own its state, the caller drives
/// `x_t` and `mask` in place. This makes it easy to inspect the trace
/// between steps and to plug alternative model logit producers in.
#[derive(Debug, Clone, Copy)]
pub struct DiffusionDecoder {
    /// Vocabulary size. Must be a positive power of two (matches
    /// the contract of [`super::super::confidence::confidence_scores`]).
    vocab: usize,
    /// Token ID used to mark a masked position. The decoder refuses
    /// to construct itself if `mask_token >= vocab`.
    mask_token: u32,
    /// Total number of denoise steps the caller intends to run.
    /// Stored so [`DiffusionDecoder::step`] can validate `step`
    /// arguments if asked to. Currently unused at runtime.
    #[allow(dead_code)]
    total_steps: usize,
    /// Re-mask strategy forwarded to every [`denoise_step`] call.
    strategy: RemaskStrategy,
    /// Confidence floor. Positions with softmax-max below this
    /// threshold are forced back to masked regardless of the remask
    /// strategy.
    confidence_threshold: f32,
}

impl DiffusionDecoder {
    /// Construct a new decoder. Validates `vocab > 0`, `vocab` is a
    /// power of two, `mask_token < vocab as u32`, `total_steps > 0`,
    /// and `confidence_threshold ∈ [0, 1]`. The remask strategy is
    /// validated on the first call to [`Self::step`].
    pub fn new(
        vocab: usize,
        mask_token: u32,
        total_steps: usize,
        strategy: RemaskStrategy,
        confidence_threshold: f32,
    ) -> Result<Self> {
        if vocab == 0 {
            return Err(KernelError::ZeroDimension {
                what: "vocab",
                got: 0,
            });
        }
        if !vocab.is_power_of_two() {
            return Err(KernelError::BadBufferLength {
                what: "vocab (must be power of two)",
                expected: vocab.next_power_of_two(),
                got: vocab,
            });
        }
        if (mask_token as usize) >= vocab {
            return Err(KernelError::OutOfRange {
                what: "mask_token",
                min: 0.0,
                max: (vocab.saturating_sub(1)) as f32,
                got: mask_token as f32,
            });
        }
        if total_steps == 0 {
            return Err(KernelError::BadBufferLength {
                what: "total_steps",
                expected: 1,
                got: 0,
            });
        }
        if !(0.0..=1.0).contains(&confidence_threshold) {
            return Err(KernelError::OutOfRange {
                what: "confidence_threshold",
                min: 0.0,
                max: 1.0,
                got: confidence_threshold,
            });
        }
        Ok(Self {
            vocab,
            mask_token,
            total_steps,
            strategy,
            confidence_threshold,
        })
    }

    /// Accessor for the configured vocabulary size.
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    /// Accessor for the configured mask token.
    pub fn mask_token(&self) -> u32 {
        self.mask_token
    }
    /// Accessor for the configured total step count.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }
    /// Accessor for the configured remask strategy.
    pub fn strategy(&self) -> RemaskStrategy {
        self.strategy
    }
    /// Accessor for the configured confidence threshold.
    pub fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    /// Run one denoise step in-place on `x_t` and `mask`.
    ///
    /// `logits` is the model's predicted row-major `[n, vocab]`
    /// logits buffer. The function calls [`denoise_step`] with the
    /// decoder's configured strategy and threshold, then computes
    /// [`DiffusionStepReport`].
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::DimMismatch`] if `mask.len() != x_t.len()`,
    /// [`KernelError::BadBufferLength`] if `logits.len() != n * vocab`,
    /// or whatever the underlying remask strategy rejects (e.g.
    /// [`KernelError::OutOfRange`] for an invalid percentile).
    pub fn step(
        &self,
        x_t: &mut Vec<u32>,
        mask: &mut Vec<bool>,
        logits: &[f32],
    ) -> Result<DiffusionStepReport> {
        let n = x_t.len();
        if mask.len() != n {
            return Err(KernelError::DimMismatch {
                what: "DiffusionDecoder::step mask vs x_t",
                expected: n,
                got: mask.len(),
            });
        }
        if logits.len() != n * self.vocab {
            return Err(KernelError::BadBufferLength {
                what: "DiffusionDecoder::step logits",
                expected: n * self.vocab,
                got: logits.len(),
            });
        }
        let update = denoise_step(
            x_t,
            mask,
            logits,
            self.strategy,
            self.confidence_threshold,
            self.vocab,
        )?;
        // Capture the new mask while replacing `*mask` in place, so we
        // can compare old-vs-new without an extra allocation.
        let new_mask = std::mem::replace(&mut *mask, update.next_mask);
        let remasked_count = mask
            .iter()
            .zip(new_mask.iter())
            .filter(|(m_new, m_old)| **m_new && !**m_old)
            .count();
        *x_t = update.next_x;
        let finished = mask.iter().all(|m| !*m);
        Ok(DiffusionStepReport {
            step: 0, // caller fills this in; we don't track step internally
            accepted_count: update.accepted_count,
            remasked_count,
            finished,
        })
    }
}
