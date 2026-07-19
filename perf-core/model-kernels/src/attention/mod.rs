//! Attention kernels for GQA, MLA, CCA, paged, tree, and dense attention.
//!
//! # Layout summary
//!
//! - `q` / `k` / `v` are flat `[seq, num_heads, head_dim]` for GQA /
//!   dense / paged.
//! - MLA uses two latent vectors (`q_latent`, `k_latent`, `v_latent`)
//!   plus two rope vectors.
//! - CCA takes a pre-compressed `compressed_k` / `compressed_v` of
//!   length `seq_k / compressed_factor`.
//! - Paged attention takes `k_cache` / `v_cache` laid out
//!   `[num_blocks, block_size, kv_heads, head_dim]` plus a
//!   `block_tables` list of `(block_id, intra_block_offset)` pairs.
//!
//! Numerical contract: floats within `1e-5` absolute or `1e-4` relative.

pub mod cca;
pub mod cca_block;
pub mod common;
pub mod dense;
pub mod gqa;
pub mod mla;
pub mod mla_cache;
pub mod paged;
pub mod tree;

pub use cca::cca_attention;
pub use cca_block::{cca_block_attend, cca_block_attend_oracle, CcaBlock};
pub use dense::dense_attention;
pub use gqa::gqa_attention;
pub use mla::mla_attention;
pub use mla_cache::{
    mla_cache_append, mla_cache_append_with_capacity, mla_cache_attend, MlaCacheEntry,
    MLA_CACHE_MAX_ENTRIES,
};
pub use paged::paged_attention;
pub use tree::tree_attention_step;
