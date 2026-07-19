//! DeepSeek-V3 style Multi-Latent Attention (MLA) cache layout.
//!
//! The classic MLA kernel in [`super::mla`] operates on the full
//! `[seq_k, d_latent]` / `[seq_k, d_rope]` pair per decode step. At
//! inference time the DeepSeek-V3 paper instead materialises the
//! compressed KV cache as a single latent vector per token plus a small
//! rope vector (the "decoupled RoPE" trick), which the attention kernel
//! reads directly. This module is the canonical store layout for that
//! scheme and a tiny attention path over it.
//!
//! # Layout
//!
//! ```text
//! MlaCacheEntry {
//!     compressed_kv: Vec<f32>,   // length d_latent
//!     k_rope:        Vec<f32>,   // length d_rope
//! }
//! ```
//!
//! The cache is `Vec<MlaCacheEntry>`; appending grows it by one
//! element. We bound the number of entries by `u32::MAX` because the
//! speculative decoder in `spec-decode` references cached positions by
//! `u32` offset.

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

/// DeepSeek-V3 attention over a flat MLA cache.
///
/// For a single query `(q_latent, q_rope)` and cache entries
/// `(compressed_kv[t], k_rope[t])`, the score against entry `t` is
///
/// ```text
/// score[t] = q_latent . compressed_kv[t] + q_rope . k_rope[t]
/// ```
///
/// and the output is the softmax-weighted sum of `compressed_kv[t]`,
/// length `d_latent`. The rope term is absorbed into the score; it
/// does not contribute to the output projection (matching the latent
/// output channel of [`super::mla::mla_attention`]).
///
/// `out` must be exactly `d_latent` long.
pub fn mla_cache_attend(
    q_latent: &[f32],
    q_rope: &[f32],
    cache: &[MlaCacheEntry],
    d_latent: usize,
    d_rope: usize,
    out: &mut [f32],
) -> Result<()> {
    if d_latent == 0 {
        return Err(KernelError::ZeroDimension {
            what: "d_latent",
            got: 0,
        });
    }
    if d_rope == 0 {
        return Err(KernelError::ZeroDimension {
            what: "d_rope",
            got: 0,
        });
    }
    if q_latent.len() != d_latent {
        return Err(KernelError::BadBufferLength {
            what: "q_latent",
            expected: d_latent,
            got: q_latent.len(),
        });
    }
    if q_rope.len() != d_rope {
        return Err(KernelError::BadBufferLength {
            what: "q_rope",
            expected: d_rope,
            got: q_rope.len(),
        });
    }
    if out.len() != d_latent {
        return Err(KernelError::BadBufferLength {
            what: "out",
            expected: d_latent,
            got: out.len(),
        });
    }
    for entry in cache {
        if entry.compressed_kv.len() != d_latent {
            return Err(KernelError::BadBufferLength {
                what: "cache.compressed_kv",
                expected: d_latent,
                got: entry.compressed_kv.len(),
            });
        }
        if entry.k_rope.len() != d_rope {
            return Err(KernelError::BadBufferLength {
                what: "cache.k_rope",
                expected: d_rope,
                got: entry.k_rope.len(),
            });
        }
    }

    // Empty cache -> zero output.
    if cache.is_empty() {
        for slot in out.iter_mut() {
            *slot = 0.0;
        }
        return Ok(());
    }

    // Compute raw scores + max.
    let mut scores: Vec<f32> = Vec::with_capacity(cache.len());
    let mut max = f32::NEG_INFINITY;
    for entry in cache {
        let mut dot_l = 0.0f32;
        for d in 0..d_latent {
            dot_l += q_latent[d] * entry.compressed_kv[d];
        }
        let mut dot_r = 0.0f32;
        for d in 0..d_rope {
            dot_r += q_rope[d] * entry.k_rope[d];
        }
        let s = dot_l + dot_r;
        scores.push(s);
        if s > max {
            max = s;
        }
    }

    // Stable softmax.
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for s in scores.iter_mut() {
        *s *= inv;
    }

    // Weighted sum of compressed_kv (latent output only).
    for slot in out.iter_mut() {
        *slot = 0.0;
    }
    for (t, entry) in cache.iter().enumerate() {
        let p = scores[t];
        for d in 0..d_latent {
            out[d] += p * entry.compressed_kv[d];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::mla_attention;
    use crate::common::Lcg;

    /// Deterministic seeded buffer of signed f32s.
    fn det(n: usize, salt: u64) -> Vec<f32> {
        let mut rng = Lcg::new(0xCAFE_BABE ^ salt);
        (0..n).map(|_| rng.next_signed()).collect()
    }

    #[test]
    fn rejects_zero_d_latent() {
        let mut cache: Vec<MlaCacheEntry> = Vec::new();
        let err = mla_cache_append(&mut cache, &[], &[0.0]).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn rejects_zero_d_rope() {
        let mut cache: Vec<MlaCacheEntry> = Vec::new();
        let err = mla_cache_append(&mut cache, &[0.0], &[]).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn rejects_inconsistent_k_rope_length() {
        let mut cache: Vec<MlaCacheEntry> = Vec::new();
        mla_cache_append(&mut cache, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        let err = mla_cache_append(&mut cache, &[1.0, 2.0], &[3.0]).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn capacity_overflow_returns_bad_buffer_length() {
        // Fake the cap: pretend the cache can hold at most 2 entries,
        // fill it, then assert the next append fails with
        // `BadBufferLength` rather than panicking or overflowing.
        let mut cache: Vec<MlaCacheEntry> = Vec::new();
        mla_cache_append_with_capacity(&mut cache, &[1.0, 2.0], &[3.0, 4.0], 2).unwrap();
        mla_cache_append_with_capacity(&mut cache, &[5.0, 6.0], &[7.0, 8.0], 2).unwrap();
        let err = mla_cache_append_with_capacity(
            &mut cache,
            &[9.0, 10.0],
            &[11.0, 12.0],
            2,
        )
        .unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn empty_cache_produces_zero_output() {
        let q_latent = [0.1f32, 0.2];
        let q_rope = [0.3f32, 0.4];
        let mut out = [99.0f32, 99.0];
        mla_cache_attend(&q_latent, &q_rope, &[], 2, 2, &mut out).unwrap();
        assert_eq!(out, [0.0, 0.0]);
    }

    /// Equivalence against the full MLA oracle: building a cache where
    /// each entry's `compressed_kv` plays the role of `v_latent`, then
    /// attending, must produce the same latent-channel output as
    /// `mla_attention` called with `k_latent = compressed_kv` and
    /// `v_latent = compressed_kv` (the absorbing-decomposition
    /// equivalent). Uses two random seeds for breadth.
    #[test]
    fn mla_cache_equivalence_against_oracle() {
        for (d_latent, d_rope, seq_k, salt) in [
            (4usize, 4usize, 6usize, 0xA1u64),
            (6, 4, 8, 0xB1),
        ] {
            let k_latent_full = det(seq_k * d_latent, salt);
            let v_latent_full = k_latent_full.clone();
            let q_latent = det(d_latent, salt.wrapping_add(1));
            let q_rope = det(d_rope, salt.wrapping_add(2));
            let k_rope_full = det(seq_k * d_rope, salt.wrapping_add(3));

            let mut full_out = vec![0.0f32; d_latent + d_rope];
            mla_attention(
                &q_latent,
                &k_latent_full,
                &v_latent_full,
                &q_rope,
                &k_rope_full,
                d_latent,
                d_rope,
                1,
                seq_k,
                &mut full_out,
            )
            .unwrap();

            let mut cache: Vec<MlaCacheEntry> = Vec::new();
            for t in 0..seq_k {
                let ck = &k_latent_full[t * d_latent..t * d_latent + d_latent];
                let kr = &k_rope_full[t * d_rope..t * d_rope + d_rope];
                mla_cache_append(&mut cache, ck, kr).unwrap();
            }

            let mut cache_out = vec![0.0f32; d_latent];
            mla_cache_attend(
                &q_latent,
                &q_rope,
                &cache,
                d_latent,
                d_rope,
                &mut cache_out,
            )
            .unwrap();

            assert_eq!(cache_out.len(), d_latent);
            for d in 0..d_latent {
                assert!(
                    crate::common::approx_eq_tol(
                        cache_out[d],
                        full_out[d],
                        1e-5,
                        1e-4,
                    ),
                    "salt=0x{salt:x} channel {d}: cache {} vs oracle {}",
                    cache_out[d],
                    full_out[d]
                );
            }
        }
    }
}
