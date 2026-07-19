//! `model-kernels` — focused, pure-Rust kernel packages per model family.
//!
//! This crate is the executable companion to [`model_plan`]. Every family
//! lives in its own module and ships:
//!
//! 1. a **scalar oracle** (`*_oracle`) used as the correctness reference
//!    for the focused Rust kernel sitting next to it;
//! 2. an **optimized Rust kernel** that the kernel registry can later
//!    dispatch on shape signatures.
//!
//! # Conventions
//!
//! - All functions are pure: no globals, no I/O, no FFI.
//! - Every public fallible API returns `Result<T, KernelError>`.
//! - Numerical tolerance for oracle comparisons is `abs = 1e-5`,
//!   `rel = 1e-4` (see [`common::approx_eq`]).
//! - Determinism: any randomness is driven by [`common::Lcg`] from a
//!   caller-supplied `u64` seed.
//!
//! # Layout
//!
//! Top-level single-file facades (matching the implementation plan):
//!
//! - [`attention_facade`]: GQA, MLA, CCA, paged, tree, and dense attention.
//! - [`moe_facade`]: router, dispatch, grouped GEMM, shared experts, reduction.
//! - [`recurrent_facade`]: DeltaNet, short convolutions, Mamba scan, RWKV.
//! - [`diffusion_facade`]: LLaDA / Dream parallel denoise and remask.
//!
//! Implementation submodules live under [`attention`], [`moe`],
//! [`recurrent`], [`diffusion`], [`quantized`] directories as focused
//! files kept under the 500-line cap.
//!
//! - [`error`]: typed kernel errors.
//! - [`common`]: tolerances, deterministic PRNG, softmax.

#![deny(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod attention;
pub mod common;
pub mod diffusion;
pub mod error;
pub mod moe;
pub mod quantized;
pub mod recurrent;
pub mod speculative;

// Spec'd single-file facades: re-export each model's public API from a
// top-level module whose name matches the task spec (`attention.rs`,
// `moe.rs`, `recurrent.rs`, `diffusion.rs`). The actual implementations
// live in the corresponding directories as focused submodules; the
// facade modules exist so callers can do e.g.
// `use model_kernels::attention::dense_attention;` directly.
pub mod attention_facade;
pub mod diffusion_facade;
pub mod moe_facade;
pub mod recurrent_facade;

pub use error::{KernelError, Result};

/// One-line kernel tag enum used by the crate's own logging. Mirrors the
/// small subset of `model_plan::operator::OperatorKind` that the kernels
/// in this crate actually dispatch to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelOp {
    /// Dense (vanilla) attention.
    DenseAttention,
    /// Grouped-query attention.
    GqaAttention,
    /// Multi-latent attention.
    MlaAttention,
    /// Compressed-context attention (ZAYA-style).
    CcaAttention,
    /// Paged attention.
    PagedAttention,
    /// Tree-shaped causal attention.
    TreeAttention,
    /// Mixture-of-experts router.
    MoeRouter,
    /// Mixture-of-experts dispatch + grouped GEMM.
    MoeDispatch,
    /// Mixture-of-experts weighted reduction.
    MoeReduce,
    /// Shared-expert dense matmul.
    MoeShared,
    /// DeltaNet chunked linear-recurrent update.
    DeltaNet,
    /// LFM2-style gated short convolution.
    ShortConv,
    /// Selective state-space scan (Mamba).
    MambaScan,
    /// RWKV time-mixing update.
    RwkvTimeMix,
    /// LLaDA / Dream parallel denoise step.
    Denoise,
    /// Remask scheduling (pure).
    Remask,
    /// Ternary pack/unpack.
    TernaryPack,
    /// Sub-byte pack/unpack.
    SubBytePack,
    /// Multi-token-prediction proposal (DeepSeek-V3 / EAGLE-style).
    SpeculativeMtp,
}

impl KernelOp {
    /// Short lowercase tag.
    pub fn tag(&self) -> &'static str {
        match self {
            KernelOp::DenseAttention => "dense_attention",
            KernelOp::GqaAttention => "gqa_attention",
            KernelOp::MlaAttention => "mla_attention",
            KernelOp::CcaAttention => "cca_attention",
            KernelOp::PagedAttention => "paged_attention",
            KernelOp::TreeAttention => "tree_attention",
            KernelOp::MoeRouter => "moe_router",
            KernelOp::MoeDispatch => "moe_dispatch",
            KernelOp::MoeReduce => "moe_reduce",
            KernelOp::MoeShared => "moe_shared",
            KernelOp::DeltaNet => "deltanet",
            KernelOp::ShortConv => "short_conv",
            KernelOp::MambaScan => "mamba_scan",
            KernelOp::RwkvTimeMix => "rwkv_time_mix",
            KernelOp::Denoise => "denoise",
            KernelOp::Remask => "remask",
            KernelOp::TernaryPack => "ternary_pack",
            KernelOp::SubBytePack => "subbyte_pack",
            KernelOp::SpeculativeMtp => "speculative_mtp",
        }
    }
}
