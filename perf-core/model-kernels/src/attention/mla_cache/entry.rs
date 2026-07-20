//! `MlaCacheEntry` layout and capacity bound.
//!
//! See the parent module docs for the DeepSeek-V3 MLA cache layout and
//! the role of this entry type.

use crate::error::{KernelError, Result};

/// Maximum number of entries the MLA cache can hold.
///
/// The speculative decoder references cache positions by `u32`; we
/// therefore refuse to append beyond `u32::MAX as usize` entries.
pub const MLA_CACHE_MAX_ENTRIES: usize = u32::MAX as usize;

/// One entry of the DeepSeek-V3 MLA cache.
///
/// `compressed_kv` carries the fused K/V latent of length `d_latent`,
/// `k_rope` carries the decoupled rope vector of length `d_rope`.
#[derive(Debug, Clone, PartialEq)]
pub struct MlaCacheEntry {
    /// Compressed K/V latent (`[d_latent]`).
    pub compressed_kv: Vec<f32>,
    /// Decoupled rope vector (`[d_rope]`).
    pub k_rope: Vec<f32>,
}

impl MlaCacheEntry {
    /// Build a new entry from raw vectors after validating their shapes.
    pub fn new(compressed_kv: Vec<f32>, k_rope: Vec<f32>) -> Result<Self> {
        if compressed_kv.is_empty() {
            return Err(KernelError::ZeroDimension {
                what: "d_latent",
                got: 0,
            });
        }
        if k_rope.is_empty() {
            return Err(KernelError::ZeroDimension {
                what: "d_rope",
                got: 0,
            });
        }
        Ok(Self {
            compressed_kv,
            k_rope,
        })
    }
}
