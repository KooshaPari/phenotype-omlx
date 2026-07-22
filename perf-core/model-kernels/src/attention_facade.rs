//! Attention kernels — single-file facade (spec: `attention.rs`).
//!
//! Re-exports the attention families implemented under `attention/`:
//!
//! - [`dense_attention`] — vanilla scaled dot-product over `[seq, head, d]`
//!   tensors.
//! - [`gqa_attention`] — grouped-query attention (multiple q-heads per
//!   kv-head).
//! - [`mla_attention`] — multi-latent attention (DeepSeek-style) operating
//!   on `q_latent` / `k_latent` / `v_latent` plus rope vectors.
//! - [`cca_attention`] — compressed-context attention (ZAYA-style) over a
//!   pre-compressed KV cache.
//! - [`paged_attention`] — paged KV cache lookup, useful for inference.
//! - [`tree_attention_step`] — tree-shaped causal attention (prefix + tree
//!   fan-out).
//!
//! Every entry point takes `[f32]` slices laid out as
//! `[seq, num_heads, head_dim]` (with shape parameters passed alongside)
//! and writes into a caller-provided `out` buffer. No FFI, no globals,
//! no allocations beyond the `out` buffer.
//!
//! Numerical tolerance: `abs = 1e-5` or `rel = 1e-4` (see
//! [`crate::common::approx_eq`]).

pub use crate::attention::cca::cca_attention;
pub use crate::attention::dense::dense_attention;
pub use crate::attention::gqa::gqa_attention;
pub use crate::attention::mla::mla_attention;
pub use crate::attention::paged::paged_attention;
pub use crate::attention::sliding_window::sliding_window_attention;
pub use crate::attention::tree::tree_attention_step;
