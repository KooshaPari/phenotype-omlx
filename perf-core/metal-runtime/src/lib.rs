//! `metal-runtime` — Metal-only pipeline that takes a `model-plan`
//! [`ModelPlan`] and produces an executable compiled pipeline, with cache,
//! bounded compilation, device fingerprinting, and deterministic fallback
//! when Metal is unavailable.
//!
//! # Public surface
//!
//! - [`DeviceFingerprint`] / [`GpuFamily`] — device description and
//!   stable, host-independent hash used as the cache-key dimension.
//! - [`PipelineCache`] / [`CompiledPipeline`] / [`EvictionPolicy`] /
//!   [`CacheStats`] — bounded LRU / FIFO cache of compiled pipelines,
//!   with optional JSON persistence via [`PipelineCache::write_through`]
//!   and [`PipelineCache::load_from_disk`].
//! - [`BoundedCompiler`] / [`CompileBudget`] / [`CompileError`] —
//!   shader-source and wall-clock compile budgets; failures are reported
//!   with both dimensions populated.
//! - [`Pipeline`] / [`StepOutput`] — the user-facing runtime handle. A
//!   pipeline is compiled once, then stepped repeatedly.
//!
//! # Compile-time guarantees
//!
//! - `#![deny(unsafe_code)]`.
//! - No `metal` crate dependency in the default build; the crate compiles
//!   and tests on Linux CI, macOS, and Windows. A future `metal` feature
//!   will add the SDK dependency and real codegen.
//! - All public types are `Sync + Send` where reasonable.
//!
//! # Module layout
//!
//! - [`fingerprint`]: [`DeviceFingerprint`] and [`GpuFamily`].
//! - [`cache`]: [`PipelineCache`] and [`CompiledPipeline`].
//! - [`compile`]: [`BoundedCompiler`] and [`CompileBudget`].
//! - [`pipeline`]: [`Pipeline`] and [`StepOutput`].
//! - [`error`]: [`CompileError`] and [`PipelineError`].

#![cfg_attr(not(all(feature = "metal", target_os = "macos")), deny(unsafe_code))]

pub mod artifact;
pub mod adaln;
pub mod cache;
pub mod compile;
pub mod dispatch;
pub mod error;
pub mod fingerprint;
pub mod joint_attention;
pub mod moe;
pub mod pipeline;
pub mod rope3d;
pub mod ternary;
pub mod temporal_attention;

pub use artifact::{
    ArtifactAllowlist, ArtifactError, MetallibArtifact, MetallibLoader, RuntimeMode,
};
pub use adaln::AdaLnError;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use adaln::adaln_rms_metal;
pub use cache::{CacheKey, CacheStats, CompiledPipeline, EvictionPolicy, PipelineCache};
pub use compile::{BoundedCompiler, CompileBudget};
pub use error::{CompileError, PipelineError};
pub use fingerprint::{DeviceFingerprint, FingerprintError, GpuFamily};
pub use joint_attention::JointAttentionError;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use joint_attention::joint_attention_metal;
pub use moe::{MoeRouter, MoeRouterError, MoeRouterOutput, MoeShape};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use moe::grouped_gemm_metal;
pub use pipeline::{Pipeline, StepOutput};
pub use rope3d::Rope3dError;
pub use ternary::TernaryGemmError;
pub use temporal_attention::TemporalAttentionError;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use rope3d::rope_3d_metal;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use ternary::ternary_gemm_metal;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use temporal_attention::temporal_window_attention_metal;

// ---------------------------------------------------------------------------
// Internal notes. `compile::plan_revision` is `pub(crate)` and is consumed
// directly by the `pipeline` module via `crate::compile::plan_revision`.
// No crate-root re-export is needed.
// ---------------------------------------------------------------------------
