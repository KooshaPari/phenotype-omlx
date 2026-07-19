//! Diffusion-family kernels — single-file facade (spec: `diffusion.rs`).
//!
//! Re-exports the diffusion primitives implemented under `diffusion/`:
//!
//! - [`confidence_scores`] — softmax-max-per-token confidence used to
//!   decide which tokens to commit.
//! - [`remask`] — apply a [`RemaskStrategy`] to a current mask
//!   (`LowConfidence { percentile }`, `EntropyBased`, `RandomFraction`, or
//!   `None`).
//! - [`denoise_step`] / [`denoise_step_sequential`] — single LLaDA / Dream
//!   parallel-denoise step producing a [`DenoiseUpdate`].
//!
//! Every entry point is pure and deterministic given its inputs. The
//! `RandomFraction` strategy uses an internal seeded LCG so test
//! expectations match across runs.

pub use crate::diffusion::confidence::confidence_scores;
pub use crate::diffusion::denoise::{
    denoise_step, denoise_step_sequential, DenoiseUpdate, RemaskStrategy,
};
pub use crate::diffusion::remask::remask;
