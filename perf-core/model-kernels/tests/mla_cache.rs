//! Integration tests for the DeepSeek-V3 MLA cache layout and the
//! multi-token-prediction (MTP) proposal kernel.
//!
//! These tests live in `tests/` (separate test crate) to exercise the
//! public surface of `model_kernels` exactly as a downstream consumer
//! (e.g. `spec-decode`) would.

use model_kernels::attention::{
    mla_attention, mla_cache_append, mla_cache_append_with_capacity, mla_cache_attend,
    MlaCacheEntry,
};
use model_kernels::common::{approx_eq_tol, Lcg};
use model_kernels::error::KernelError;
use model_kernels::speculative::{mtp_propose, mtp_verify};

const SEED: u64 = 0xCAFE_BABE;

/// Deterministic signed f32 buffer of length `n`.
fn det(n: usize, salt: u64) -> Vec<f32> {
    let mut rng = Lcg::new(SEED ^ salt);
    (0..n).map(|_| rng.next_signed()).collect()
}

// ---------------------------------------------------------------------------
// MLA cache round-trip
// ---------------------------------------------------------------------------

#[test]
fn mla_cache_round_trip_matches_mla_attention_oracle() {
    let d_latent = 4;
    let d_rope = 4;
    let seq_k = 8;
    let n_entries = 8;

    let q_latent = det(d_latent, 0xC0);
    let q_rope = det(d_rope, 0xC1);

    // Hand-rolled latents so we can assert exact values.
    let mut cache: Vec<MlaCacheEntry> = Vec::new();
    let mut k_latent_full = Vec::with_capacity(seq_k * d_latent);
    let mut k_rope_full = Vec::with_capacity(seq_k * d_rope);
    let mut v_latent_full = Vec::with_capacity(seq_k * d_latent);
    for i in 0..n_entries {
        let ck: Vec<f32> = (0..d_latent)
            .map(|d| 0.1 * (i * d_latent + d) as f32)
            .collect();
        let kr: Vec<f32> = (0..d_rope)
            .map(|d| 0.05 * (i * d_rope + d) as f32)
            .collect();
        mla_cache_append(&mut cache, &ck, &kr).unwrap();
        k_latent_full.extend_from_slice(&ck);
        k_rope_full.extend_from_slice(&kr);
        v_latent_full.extend_from_slice(&ck);
    }
    assert_eq!(cache.len(), n_entries);

    // Reference: mla_attention with the same latents.
    let mut oracle = vec![0.0f32; d_latent + d_rope];
    mla_attention(
        &q_latent,
        &k_latent_full,
        &v_latent_full,
        &q_rope,
        &k_rope_full,
        d_latent,
        d_rope,
        1,
        n_entries,
        &mut oracle,
    )
    .unwrap();

    let mut cache_out = vec![0.0f32; d_latent];
    mla_cache_attend(&q_latent, &q_rope, &cache, d_latent, d_rope, &mut cache_out).unwrap();

    assert_eq!(cache_out.len(), d_latent);
    for d in 0..d_latent {
        assert!(
            approx_eq_tol(cache_out[d], oracle[d], 1e-5, 1e-4),
            "channel {d}: cache {} vs oracle {}",
            cache_out[d],
            oracle[d]
        );
    }
}

#[test]
fn mla_cache_append_beyond_capacity_returns_bad_buffer_length() {
    // We can't allocate u32::MAX entries; use the with_capacity variant
    // to exercise the exact same overflow branch without the full
    // memory cost.
    let mut cache: Vec<MlaCacheEntry> = Vec::new();
    mla_cache_append_with_capacity(&mut cache, &[1.0, 2.0], &[3.0, 4.0], 1).unwrap();
    let err = mla_cache_append_with_capacity(&mut cache, &[5.0, 6.0], &[7.0, 8.0], 1).unwrap_err();
    match err {
        KernelError::BadBufferLength {
            what,
            expected,
            got,
        } => {
            assert_eq!(what, "mla_cache.len");
            assert_eq!(expected, 1);
            assert_eq!(got, 2);
        }
        other => panic!("expected BadBufferLength, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// MTP propose / verify
// ---------------------------------------------------------------------------

#[test]
fn mtp_propose_returns_argmax_per_offset() {
    // Two draft rows, vocab = 5. Row 0 has its max at index 4;
    // row 1 has its max at index 1.
    let logits = vec![
        0.1f32, 0.2, 0.3, 0.4, 9.0, // row 0
        0.5, 7.0, 0.6, 0.7, 0.8, // row 1
    ];
    let p = mtp_propose(&logits, &[0, 1], 5).unwrap();
    assert_eq!(p.tokens, vec![4, 1]);
    assert_eq!(p.accepted_mask.len(), 2);
    assert!(p.accepted_mask.iter().all(|&m| !m));
}

#[test]
fn mtp_propose_with_reordered_offsets() {
    let logits = vec![
        0.0f32, 0.0, 0.0, 1.0, // row 0 -> token 3
        5.0, 0.0, 0.0, 0.0, // row 1 -> token 0
        0.0, 2.0, 0.0, 0.0, // row 2 -> token 1
    ];
    // Re-ordered draft offsets [2, 0, 1].
    let p = mtp_propose(&logits, &[2, 0, 1], 4).unwrap();
    assert_eq!(p.tokens, vec![1, 3, 0]);
}

#[test]
fn mtp_verify_accepts_all_ones_high_confidence() {
    // Seed a 2-step proposal; verifier logits are dominated by token 0
    // for row 0 and token 1 for row 1.
    let proposal = model_kernels::speculative::MtpProposal {
        tokens: vec![0, 1],
        accepted_mask: vec![false, false],
    };
    let verifier = vec![
        // row 0: massive mass on token 0
        100.0f32, -100.0, -100.0, // row 1: massive mass on token 1
        -100.0, 100.0, -100.0,
    ];
    let mask = mtp_verify(&proposal, &verifier, 3, 0.9).unwrap();
    assert_eq!(mask, vec![true, true]);
}

#[test]
fn mtp_verify_rejects_all_zeros_low_confidence() {
    // Same proposal; uniform verifier logits -> prob = 1/vocab = 1/3 ≈ 0.33.
    let proposal = model_kernels::speculative::MtpProposal {
        tokens: vec![0, 1],
        accepted_mask: vec![false, false],
    };
    let verifier = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mask = mtp_verify(&proposal, &verifier, 3, 0.9).unwrap();
    assert_eq!(mask, vec![false, false]);
}

#[test]
fn mtp_verify_threshold_split_decision() {
    // Row 0 is uniform -> 1/2 = 0.5; threshold 0.6 -> rejected (proposal token 0).
    // Row 1 strongly favours token 1 (logit 10 vs 0) -> softmax ≈ [~0, ~1.0] -> accepted.
    let proposal = model_kernels::speculative::MtpProposal {
        tokens: vec![0, 1],
        accepted_mask: vec![false, false],
    };
    let verifier = vec![0.0f32, 0.0, 0.0, 10.0];
    let mask = mtp_verify(&proposal, &verifier, 2, 0.6).unwrap();
    assert_eq!(mask, vec![false, true]);
}
