//! Append operations for the MLA cache.
//!
//! See the parent module docs for the DeepSeek-V3 MLA cache layout and
//! the validation rules enforced on every append.

use crate::error::{KernelError, Result};

use super::entry::{MlaCacheEntry, MLA_CACHE_MAX_ENTRIES};

/// Append a new MLA cache entry after validating the latent and rope
/// shapes match the previously-observed dimensions.
///
/// The first call establishes the expected `d_latent` and `d_rope`.
/// Subsequent calls must agree with the established dimensions and
/// with each other (the rope and latent vectors must have the same
/// length they had on the first append).
///
/// Returns `Err(BadBufferLength)` if appending would exceed
/// [`MLA_CACHE_MAX_ENTRIES`].
pub fn mla_cache_append(
    cache: &mut Vec<MlaCacheEntry>,
    compressed_kv: &[f32],
    k_rope: &[f32],
) -> Result<()> {
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
    if cache.len() >= MLA_CACHE_MAX_ENTRIES {
        return Err(KernelError::BadBufferLength {
            what: "mla_cache.len",
            expected: MLA_CACHE_MAX_ENTRIES,
            got: cache.len() + 1,
        });
    }
    // Establish / check shape on subsequent inserts.
    if let Some(first) = cache.first() {
        if compressed_kv.len() != first.compressed_kv.len() {
            return Err(KernelError::BadBufferLength {
                what: "compressed_kv",
                expected: first.compressed_kv.len(),
                got: compressed_kv.len(),
            });
        }
        if k_rope.len() != first.k_rope.len() {
            return Err(KernelError::BadBufferLength {
                what: "k_rope",
                expected: first.k_rope.len(),
                got: k_rope.len(),
            });
        }
    }
    cache.push(MlaCacheEntry {
        compressed_kv: compressed_kv.to_vec(),
        k_rope: k_rope.to_vec(),
    });
    Ok(())
}

/// Append a new MLA cache entry if and only if the cache would still
/// fit under `capacity` entries. Used by tests to exercise the
/// "capacity overflow" path without actually allocating `u32::MAX`
/// `MlaCacheEntry`s.
///
/// Behaves like [`mla_cache_append`] but reports the offending length
/// against the supplied `capacity` rather than [`MLA_CACHE_MAX_ENTRIES`]
/// when the cache is already at or beyond that capacity.
pub fn mla_cache_append_with_capacity(
    cache: &mut Vec<MlaCacheEntry>,
    compressed_kv: &[f32],
    k_rope: &[f32],
    capacity: usize,
) -> Result<()> {
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
    if cache.len() >= capacity {
        return Err(KernelError::BadBufferLength {
            what: "mla_cache.len",
            expected: capacity,
            got: cache.len() + 1,
        });
    }
    if let Some(first) = cache.first() {
        if compressed_kv.len() != first.compressed_kv.len() {
            return Err(KernelError::BadBufferLength {
                what: "compressed_kv",
                expected: first.compressed_kv.len(),
                got: compressed_kv.len(),
            });
        }
        if k_rope.len() != first.k_rope.len() {
            return Err(KernelError::BadBufferLength {
                what: "k_rope",
                expected: first.k_rope.len(),
                got: k_rope.len(),
            });
        }
    }
    cache.push(MlaCacheEntry {
        compressed_kv: compressed_kv.to_vec(),
        k_rope: k_rope.to_vec(),
    });
    Ok(())
}
