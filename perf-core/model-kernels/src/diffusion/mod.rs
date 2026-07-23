//! Diffusion-family kernels: DiffusionGemma-oriented parallel denoise, with
//! LLaDA/Dream retained as deterministic regression fixtures.
//!
//! Every function in this module is pure and deterministic. Random
//! re-mask policies accept an optional seed via [`RemaskStrategy`]
//! extensions in a future revision; for now the `RandomFraction`
//! strategy uses a deterministic LCG so tests are reproducible.

pub mod confidence;
pub mod decoder;
pub mod denoise;
pub mod flow;
pub mod remask;

pub use confidence::confidence_scores;
pub use decoder::{DiffusionDecoder, DiffusionStepReport};
pub use denoise::{denoise_step, denoise_step_sequential, DenoiseUpdate, RemaskStrategy};
pub use flow::{classifier_free_guidance, flow_sigma_schedule};
pub use remask::remask;
