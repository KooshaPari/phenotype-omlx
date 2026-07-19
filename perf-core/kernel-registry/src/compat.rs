//! Minimal compatibility mirror of the `model-plan` types consumed by
//! `kernel-registry`.
//!
//! `model-plan` is being created in parallel by a separate subagent (Task 2).
//! Until that crate lands a stable public surface, `kernel-registry` declares
//! its own copy of the small subset of types it needs. When `model-plan` is
//! stable, the next pass should:
//!
//! 1. Add `model-plan` as a `kernel-registry` dependency.
//! 2. Replace every `compat::TypeName` with `model_plan::TypeName`.
//! 3. Delete this module.
//!
//! Every type in this module is intentionally `#[non_exhaustive]` (where
//! supported) and uses stable serde representations so the de-duplication is
//! purely mechanical.

use serde::{Deserialize, Serialize};

/// Operator kind carried by [`crate::KernelKey::operator_kind`].
///
/// Mirrors `model_plan::operator::OperatorKind`. Only the subset required by
/// the kernel registry is enumerated; additional variants are added without
/// breaking existing serialized form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OperatorKind {
    DenseMatmul,
    GroupedMatmul,
    Attention,
    /// Grouped-query attention.
    Gqa,
    /// Multi-head latent attention.
    Mla,
    /// Compressed cross attention.
    Cca,
    TreeAttention,
    PagedAttention,
    /// Mixture-of-experts routing or expert GEMM.
    Moe,
    /// Shared expert path within an MoE block.
    MoeSharedExpert,
    /// DeltaNet / linear-recurrent update.
    DeltaNet,
    ShortConv,
    Scan,
    /// Recurrent state update (RNN-style carry).
    Recurrent,
    /// Diffusion denoise or remask step.
    Diffusion,
    /// Discrete (masked) diffusion language-model step (MDLM / D3PM / SEDD).
    /// Differs from [`OperatorKind::Diffusion`] in that the noising
    /// process replaces tokens with a dedicated `[MASK]` token rather
    /// than adding Gaussian noise; the denoising step decodes one
    /// masked position per call and re-masks a fraction determined by
    /// a noise schedule (linear or cosine).
    DiscreteDiffusion,
    /// Speculative proposal / verification path.
    Speculative,
    /// Sub-byte or ternary encode/decode and fused compute.
    Quantized,
    /// Marker for unrecognized operator names; preserves forward compatibility
    /// without panicking on lookup.
    Unknown,
}

/// Attention variant carried by [`crate::KernelKey::attention_kind`]. Only
/// meaningful when `operator_kind == AttentionKind::*`; serialized as
/// `Option<AttentionKind>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AttentionKind {
    Gqa,
    Mla,
    Cca,
    Tree,
    Paged,
    Standard,
}

/// Quantization policy attached to [`crate::KernelKey::quantization`].
///
/// Mirrors `model_plan::quantization::QuantizationPolicy`. The `Unknown`
/// variant preserves forward compatibility for plan files that reference
/// quantization schemes added after this crate was last built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum QuantizationPolicy {
    None,
    Fp8,
    Int8,
    Int4,
    /// Ternary (1.58-bit) weights, paired with activations.
    Ternary,
    SubByte,
    Unknown,
}

/// DType mirrored from `model_plan::dtype::DType`. Used both in
/// [`crate::KernelKey::dtype`] and in `Candidate::supports_dtypes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DType {
    Fp32,
    Fp16,
    Bf16,
    Fp8,
    Int8,
    Int4,
    Bool,
    Unknown,
}