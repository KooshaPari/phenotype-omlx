//! Attention path over a flat MLA cache.
//!
//! See the parent module docs for the DeepSeek-V3 MLA cache layout and
//! the score/output formula implemented by [`mla_cache_attend`].

use crate::error::{KernelError, Result};

use super::entry::MlaCacheEntry;

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
/// output channel of [`super::super::mla::mla_attention`]).
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
