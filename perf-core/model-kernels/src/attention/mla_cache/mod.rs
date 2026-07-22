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

mod append;
mod attend;
mod entry;

pub use append::{mla_cache_append, mla_cache_append_with_capacity};
pub use attend::mla_cache_attend;
pub use entry::{MlaCacheEntry, MLA_CACHE_MAX_ENTRIES};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::mla_attention;
    use crate::common::Lcg;
    use crate::error::KernelError;

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
