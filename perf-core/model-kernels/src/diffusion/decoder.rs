//! Multi-step diffusion decoder: LLaDA / Dream acceptance trace.
//!
//! This module owns the [`DiffusionDecoder`] orchestrator that drives a
//! multi-step trace on top of the pure [`super::denoise::denoise_step`]
//! kernel. The decoder is the canonical entry point exercised by the
//! acceptance matrix in `02_SPECIFICATIONS.md` for the LLaDA and Dream
//! model families (parallel denoise, confidence/remask scheduling,
//! variable active set).
//!
//! All randomness — where present — flows through [`crate::common::Lcg`]
//! from a caller-supplied `u64` seed, so a trace is fully deterministic
//! given the same seed and the same logits sequence.

use crate::error::{KernelError, Result};

use super::denoise::{denoise_step, RemaskStrategy};

/// Report of one step of a diffusion decoder trace.
///
/// Returned by [`DiffusionDecoder::step`]. The caller drives the outer
/// loop: the decoder does not run for a fixed number of internal steps,
/// it just exposes one step at a time so callers can inspect the
/// intermediate state.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionStepReport {
    /// 0-indexed step number (the order in which the caller invoked
    /// [`DiffusionDecoder::step`]).
    pub step: usize,
    /// Number of positions that are *not* masked after the step
    /// (i.e. net newly-accepted positions after remask).
    pub accepted_count: usize,
    /// Number of positions that were re-masked during the step
    /// (i.e. transitioned from `false -> true` between input mask and
    /// output mask).
    pub remasked_count: usize,
    /// `true` iff every position is unmasked (`mask[i] == false` for
    /// all `i`). The trace has finished when this is `true`.
    pub finished: bool,
}

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
    /// the contract of [`super::confidence::confidence_scores`]).
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
            return Err(KernelError::ZeroDimension { what: "vocab", got: 0 });
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

// ---------------------------------------------------------------------------
// Oracle tests (TDD)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Lcg;

    /// Build a logits row that yields softmax-max = 0.5 for `vocab=8`.
    /// logit x at the argmax index with e^x / (e^x + (vocab - 1)) = 0.5
    /// → e^x = (vocab - 1) → x = ln(vocab - 1).
    fn uniform_half_confidence_logits(n: usize, vocab: usize, argmax: usize) -> Vec<f32> {
        assert!(vocab > 1);
        assert!(argmax < vocab);
        let x = ((vocab - 1) as f64).ln() as f32;
        let mut out = Vec::with_capacity(n * vocab);
        for _ in 0..n {
            for j in 0..vocab {
                if j == argmax {
                    out.push(x);
                } else {
                    out.push(0.0);
                }
            }
        }
        out
    }

    #[test]
    fn new_rejects_zero_vocab() {
        let err = DiffusionDecoder::new(
            0,
            0,
            4,
            RemaskStrategy::LowConfidence { percentile: 50.0 },
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn new_rejects_non_power_of_two_vocab() {
        let err = DiffusionDecoder::new(
            6, // not a power of two
            0,
            4,
            RemaskStrategy::LowConfidence { percentile: 50.0 },
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn new_rejects_mask_token_outside_vocab() {
        let err = DiffusionDecoder::new(
            8,
            8, // >= vocab
            4,
            RemaskStrategy::LowConfidence { percentile: 50.0 },
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, KernelError::OutOfRange { .. }));
    }

    #[test]
    fn new_rejects_zero_total_steps() {
        let err = DiffusionDecoder::new(
            8,
            0,
            0,
            RemaskStrategy::LowConfidence { percentile: 50.0 },
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn new_rejects_invalid_confidence_threshold() {
        let err = DiffusionDecoder::new(
            8,
            0,
            4,
            RemaskStrategy::LowConfidence { percentile: 50.0 },
            1.5,
        )
        .unwrap_err();
        assert!(matches!(err, KernelError::OutOfRange { .. }));
    }

    #[test]
    fn step_returns_error_on_mask_length_mismatch() {
        let dec = DiffusionDecoder::new(
            8,
            0,
            4,
            RemaskStrategy::LowConfidence { percentile: 50.0 },
            0.0,
        )
        .unwrap();
        let mut x_t = vec![0u32; 4];
        let mut mask = vec![true, true, true]; // mismatched
        let logits = vec![0.0f32; 4 * 8];
        let err = dec.step(&mut x_t, &mut mask, &logits).unwrap_err();
        assert!(matches!(err, KernelError::DimMismatch { .. }));
    }

    #[test]
    fn step_returns_error_on_logits_length_mismatch() {
        let dec = DiffusionDecoder::new(
            8,
            0,
            4,
            RemaskStrategy::LowConfidence { percentile: 50.0 },
            0.0,
        )
        .unwrap();
        let mut x_t = vec![0u32; 4];
        let mut mask = vec![true; 4];
        let logits = vec![0.0f32; 3 * 8]; // wrong length
        let err = dec.step(&mut x_t, &mut mask, &logits).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn oracle_low_confidence_finishes_with_unmasked_tokens() {
        // 32-token sequence, vocab=8, mask_token=0, 4 denoise steps with
        // LowConfidence { percentile: 50.0 }. The trace must finish
        // with every position unmasked AND every mask_token-initial
        // position must have a non-mask_token value.
        //
        // Construction: logits rows with softmax-max = 0.5 uniformly
        // across all 32 positions. Under the 50th-percentile remask,
        // threshold = median = 0.5 and `score < 0.5` is false for all,
        // so nothing gets re-masked. Every step is a clean decode.
        let n = 32usize;
        let vocab = 8usize;
        let mask_token = 0u32;
        let argmax_token = 7usize; // non-mask_token within vocab
        let dec = DiffusionDecoder::new(
            vocab,
            mask_token,
            4,
            RemaskStrategy::LowConfidence { percentile: 50.0 },
            0.0,
        )
        .unwrap();
        let mut x_t = vec![mask_token; n];
        let mut mask = vec![true; n];
        let logits = uniform_half_confidence_logits(n, vocab, argmax_token);
        for _step in 0..4 {
            dec.step(&mut x_t, &mut mask, &logits).unwrap();
        }
        assert!(
            mask.iter().all(|m| !*m),
            "every position must be unmasked after 4 steps"
        );
        assert!(
            x_t.iter().all(|&t| t != mask_token),
            "every mask_token-initial position must have a non-mask_token value"
        );
        assert!(x_t.iter().all(|&t| t == argmax_token as u32));
    }

    #[test]
    fn oracle_deterministic_replay_produces_identical_sequences() {
        // Two runs with the same seed must produce identical final
        // sequences. Use an LCG-driven logits generator so determinism
        // is observable (constant logits would trivially satisfy the
        // assertion).
        let n = 32usize;
        let vocab = 8usize;
        let mask_token = 0u32;
        let total_steps = 4usize;
        let dec = DiffusionDecoder::new(
            vocab,
            mask_token,
            total_steps,
            RemaskStrategy::LowConfidence { percentile: 50.0 },
            0.0,
        )
        .unwrap();

        let mut rng1 = Lcg::new(0xDEAD_BEEF);
        let mut rng2 = Lcg::new(0xDEAD_BEEF);
        let mut x1 = vec![mask_token; n];
        let mut x2 = vec![mask_token; n];
        let mut m1 = vec![true; n];
        let mut m2 = vec![true; n];
        for _ in 0..total_steps {
            let logits1: Vec<f32> = (0..n * vocab).map(|_| rng1.next_signed()).collect();
            let logits2: Vec<f32> = (0..n * vocab).map(|_| rng2.next_signed()).collect();
            assert_eq!(logits1, logits2, "LCG with same seed must match");
            dec.step(&mut x1, &mut m1, &logits1).unwrap();
            dec.step(&mut x2, &mut m2, &logits2).unwrap();
        }
        assert_eq!(x1, x2);
        assert_eq!(m1, m2);
    }

    #[test]
    fn oracle_remask_strategy_validation_invalid_percentile_returns_out_of_range() {
        // Construct a decoder with an invalid percentile and confirm
        // the error propagates out of step() as OutOfRange.
        let dec_bad = DiffusionDecoder::new(
            8,
            0,
            4,
            RemaskStrategy::LowConfidence { percentile: -10.0 },
            0.0,
        )
        .unwrap();
        let mut x_t = vec![0u32; 4];
        let mut mask = vec![true; 4];
        let logits = vec![0.0f32; 4 * 8];
        let err = dec_bad.step(&mut x_t, &mut mask, &logits).unwrap_err();
        assert!(matches!(err, KernelError::OutOfRange { .. }));
    }
}