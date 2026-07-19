//! Diffusion-family kernels: LLaDA / Dream parallel denoise.
//!
//! Every function in this module is pure and deterministic. Random
//! re-mask policies accept an optional seed via [`RemaskStrategy`]
//! extensions in a future revision; for now the `RandomFraction`
//! strategy uses a deterministic LCG so tests are reproducible.

pub mod confidence;
pub mod denoise;
pub mod remask;

pub use confidence::confidence_scores;
pub use denoise::{denoise_step, denoise_step_sequential, DenoiseUpdate, RemaskStrategy};
pub use remask::remask;
