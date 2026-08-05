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

pub mod adaln;
pub mod artifact;
pub mod cache;
pub mod cca;
pub mod compile;
pub mod deltanet;
pub mod diffusion_confidence;
pub mod diffusion_dispatch;
pub mod diffusion_dispatch_metal;
pub mod diffusion_parity;
pub mod diffusion_self_verify;
pub mod diffusion_state;
pub mod diffusion_telemetry;
pub mod dispatch;
pub mod error;
pub mod fingerprint;
pub mod flow_step;
pub mod joint_attention;
pub mod mamba;
pub mod mamba_scan;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_cache;
pub mod mla_cache;
pub mod moe;
pub mod native_catalog;
pub mod pipeline;
pub mod retnet;
pub mod rope3d;
pub mod rwkv;
pub mod short_conv;
pub mod temporal_attention;
pub mod ternary;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use adaln::adaln_rms_metal;
pub use adaln::AdaLnError;
pub use artifact::{
    ArtifactAllowlist, ArtifactError, ArtifactManifest, ArtifactManifestEntry, MetallibArtifact,
    MetallibLoader, RuntimeMode,
};
pub use cache::{CacheKey, CacheStats, CompiledPipeline, EvictionPolicy, PipelineCache};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use cca::cca_block_attend_metal;
pub use cca::CcaError;
pub use compile::{BoundedCompiler, CompileBudget};
pub use deltanet::DeltaNetError;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use deltanet::{deltanet_step_metal, deltanet_step_metal_two_pass};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use diffusion_confidence::diffusion_argmax_confidence_metal;
pub use diffusion_confidence::DiffusionConfidenceError;
pub use diffusion_dispatch::{DiffusionDispatchEvaluation, DiffusionDispatchPlan, DiffusionStage};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use diffusion_dispatch_metal::{
    diffusion_active_compact_metal, diffusion_active_compact_metal_with_telemetry,
    diffusion_remask_metal, diffusion_remask_metal_with_telemetry, diffusion_trajectory_metal,
    diffusion_trajectory_metal_with_telemetry,
};
pub use diffusion_dispatch_metal::{validate_diffusion_threshold, DiffusionDispatchError};
pub use diffusion_parity::{compare_f32, compare_u32, compare_u8, DiffusionParityError};
pub use diffusion_self_verify::{
    DiffusionSelfVerifyError, DiffusionVerificationBlock, DiffusionVerificationPlan,
};
pub use diffusion_state::{DiffusionStateLayout, DiffusionStateLayoutError};
pub use diffusion_telemetry::{
    DiffusionDispatchDecision, DiffusionDispatchReport, DiffusionDispatchTelemetry,
    DiffusionRollbackPolicy, DiffusionStageOutcome, DiffusionStageTelemetry,
    DiffusionTelemetryError,
};
pub use error::{CompileError, PipelineError};
pub use fingerprint::{DeviceFingerprint, FingerprintError, GpuFamily};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use flow_step::flow_cfg_step_metal;
pub use flow_step::FlowStepError;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use joint_attention::joint_attention_metal;
pub use joint_attention::JointAttentionError;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use mamba::mamba_selective_step_metal;
pub use mamba::MambaError;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use mamba_scan::mamba_selective_scan_metal;
pub use mamba_scan::{validate_scan_shapes, MambaScanError};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use mla_cache::mla_cache_attend_metal;
pub use mla_cache::MlaCacheError;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use moe::grouped_gemm_metal;
pub use moe::{MoeRouter, MoeRouterError, MoeRouterOutput, MoeShape};
pub use native_catalog::{
    all_specs as native_kernel_specs, spec_for_tag as native_kernel_spec, NativeKernelBinding,
    NativeKernelBundle, NativeKernelError, NativeKernelSpec,
};
pub use pipeline::{Pipeline, StepOutput};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use retnet::retnet_retention_step_metal;
pub use retnet::RetNetError;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use rope3d::rope_3d_metal;
pub use rope3d::Rope3dError;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use rwkv::rwkv7_time_mix_metal;
pub use rwkv::RwkvError;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use short_conv::short_conv1d_step_metal;
pub use short_conv::ShortConvError;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use temporal_attention::temporal_window_attention_metal;
pub use temporal_attention::TemporalAttentionError;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use ternary::ternary_gemm_metal;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use ternary::ternary_gemm_metal_from_host;
pub use ternary::TernaryGemmError;

// ---------------------------------------------------------------------------
// Internal notes. `compile::plan_revision` is `pub(crate)` and is consumed
// directly by the `pipeline` module via `crate::compile::plan_revision`.
// No crate-root re-export is needed.
// ---------------------------------------------------------------------------
