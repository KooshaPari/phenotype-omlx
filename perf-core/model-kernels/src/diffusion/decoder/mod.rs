//! Multi-step diffusion decoder: LLaDA / Dream acceptance trace.
//!
//! This module owns the [`DiffusionDecoder`] orchestrator that drives a
//! multi-step trace on top of the pure [`super::super::denoise::denoise_step`]
//! kernel. The decoder is the canonical entry point exercised by the
//! acceptance matrix in `02_SPECIFICATIONS.md` for the LLaDA and Dream
//! model families (parallel denoise, confidence/remask scheduling,
//! variable active set).
//!
//! All randomness — where present — flows through [`crate::common::Lcg`]
//! from a caller-supplied `u64` seed, so a trace is fully deterministic
//! given the same seed and the same logits sequence.

// `mod decoder;` matches the parent directory `decoder/`. Renaming the
// inner file would obscure the natural mapping (`DiffusionDecoder` ↔
// `decoder.rs`); suppress the inception lint instead.
#[allow(clippy::module_inception)]
mod decoder;
mod report;

pub use decoder::DiffusionDecoder;
pub use report::DiffusionStepReport;

#[cfg(test)]
mod tests;
