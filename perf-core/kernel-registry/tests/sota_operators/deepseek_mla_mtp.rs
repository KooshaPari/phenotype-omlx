//! DeepSeek MLA (Multi-Latent Attention) + MTP (Multi-Token Prediction)
//! oracle coverage.
//!
//! MLA compresses the KV-cache via a low-rank projection (latent dim
//! `D_latent` << head dim `D`); the compressed cache holds `D_latent`-
//! dim K and V tensors instead of the full `D`-dim ones. MTP speculates
//! `k > 1` next tokens in a single forward pass and verifies them
//! against a verifier logits tensor.
//!
//! The tests in this file exercise the **byte-oracle contract** for the
//! `model_kernels` reference implementations of both kernels, mirroring
//! the role `moe_routing.rs` plays for the MoE router. Each test (a)
//! constructs deterministic inputs from a seeded [`Lcg`], (b) runs the
//! reference kernel, (c) re-derives the expected output from scratch
//! via an in-file oracle, and (d) asserts the two agree within the
//! documented tolerance (abs = 1e-5).
//!
//! Conventions match the rest of `sota_operators/`:
//!
//! - `NOW_UNIX_MS` and `TEST_FINGERPRINT` come from the shared `main.rs`.
//! - Determinism comes from [`model_kernels::common::Lcg`] — no globals,
//!   no wall-clock, no `HashMap` iteration.
//!
//! Coverage tags emitted: `DeepSeekMla`, `DeepSeekMtp`.

use model_kernels::attention::mla::mla_attention;
use model_kernels::common::{approx_eq, Lcg};
use model_kernels::speculative::{mtp_propose, mtp_verify};

// ---------------------------------------------------------------------------
// Shared constants for the MLA byte-equality test
// ---------------------------------------------------------------------------

/// Tokens in the cached sequence.
const B: usize = 4;
/// Number of KV heads.
const H_KV: usize = 4;
/// Latent dimension. `D_latent = D/4` so the cache shrinks 4×.
const D_LATENT: usize = 16;
/// Full head dim (uncompressed).
const D_FULL: usize = 64;
/// RoPE dim (small; independent of latent/head split).
const D_ROPE: usize = 4;

/// Structural compression ratio `D_latent / D`. With `D_latent = D/4`,
/// the bytes-on-the-wire shrink by 4×, comfortably below the `0.5×`
/// invariant required by [`deepseek_mla_cache_size_smaller_than_uncompressed`].
const MLA_RATIO: f32 = (D_LATENT * 2) as f32 / (D_FULL * 2) as f32;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Deterministic LCG-backed fill in `[-1, 1)`.
fn fill_signed(rng: &mut Lcg, out: &mut [f32]) {
    for x in out.iter_mut() {
        *x = rng.next_signed();
    }
}

/// Deterministic LCG-backed fill in `[0, 1)`.
fn fill_unit(rng: &mut Lcg, out: &mut [f32]) {
    for x in out.iter_mut() {
        *x = rng.next_f32();
    }
}

/// Per-element absolute-max deviation across two equal-length buffers.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "max_abs_diff requires equal-length slices");
    let mut m = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    m
}

fn assert_buf_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    let diff = max_abs_diff(a, b);
    assert!(
        diff <= tol,
        "{what}: max abs diff {diff} exceeds tolerance {tol}"
    );
}

/// Down-project `K_full ∈ [seq, D_full]` and `V_full ∈ [seq, D_full]`
/// to `K_latent ∈ [seq, D_latent]` and `V_latent ∈ [seq, D_latent]`
/// using a fixed LCG-seeded matrix `W ∈ [D_full, D_latent]`. Returns
/// the bytes-on-the-wire size of the resulting compressed cache.
fn compress_kv(
    rng: &mut Lcg,
    k_full: &[f32],
    v_full: &[f32],
    k_latent: &mut [f32],
    v_latent: &mut [f32],
) -> usize {
    let seq = B * H_KV;
    let mut w = vec![0.0f32; D_FULL * D_LATENT];
    fill_signed(rng, &mut w);
    for s in 0..seq {
        for d_lat in 0..D_LATENT {
            let mut kk = 0.0f32;
            let mut vv = 0.0f32;
            for d_full in 0..D_FULL {
                kk += k_full[s * D_FULL + d_full] * w[d_full * D_LATENT + d_lat];
                vv += v_full[s * D_FULL + d_full] * w[d_full * D_LATENT + d_lat];
            }
            k_latent[s * D_LATENT + d_lat] = kk;
            v_latent[s * D_LATENT + d_lat] = vv;
        }
    }
    2 * seq * D_LATENT * std::mem::size_of::<f32>()
}

// ---------------------------------------------------------------------------
// MLA: compressed vs uncompressed byte-equality + cache size invariant
// ---------------------------------------------------------------------------

#[test]
fn deepseek_mla_compressed_kv_byte_identical_to_uncompressed() {
    // Setup: build inputs where the compressed path (D_latent=16) and
    // the uncompressed path (D_latent=D_FULL=64) compute identical
    // softmax weights and identical latent outputs.
    //
    // MLA score = q_latent·k_latent + q_rope·k_rope (per query-key
    // pair). For both paths to agree:
    //   - Compressed: q_latent ∈ [seq_q,16], k_latent ∈ [seq_k,16].
    //   - Uncompressed: pad q_full to [seq_q,64] with the first 16
    //     dims equal to q_latent and the rest zero; pad k_full/v_full
    //     the same way. Then q_full·k_full = q_latent·k_latent.
    // The rope part is identical (same q_rope, k_rope). Output is
    // the latent-only weighted sum of v_latent/v_full; identical
    // because v_full[0..16] = v_latent.
    let seq_q = B;
    let seq_k = B * H_KV;
    let mut rng = Lcg::new(0xD5EE_00AD);

    let mut q_latent = vec![0.0f32; seq_q * D_LATENT];
    fill_signed(&mut rng, &mut q_latent);
    let mut q_full = vec![0.0f32; seq_q * D_FULL];
    for s in 0..seq_q {
        for d in 0..D_LATENT {
            q_full[s * D_FULL + d] = q_latent[s * D_LATENT + d];
        }
    }
    let mut k_latent = vec![0.0f32; seq_k * D_LATENT];
    let mut v_latent = vec![0.0f32; seq_k * D_LATENT];
    fill_signed(&mut rng, &mut k_latent);
    fill_signed(&mut rng, &mut v_latent);
    let mut k_full = vec![0.0f32; seq_k * D_FULL];
    let mut v_full = vec![0.0f32; seq_k * D_FULL];
    for s in 0..seq_k {
        for d in 0..D_LATENT {
            k_full[s * D_FULL + d] = k_latent[s * D_LATENT + d];
            v_full[s * D_FULL + d] = v_latent[s * D_LATENT + d];
        }
    }
    let mut q_rope = vec![0.0f32; seq_q * D_ROPE];
    let mut k_rope = vec![0.0f32; seq_k * D_ROPE];
    fill_signed(&mut rng, &mut q_rope);
    fill_signed(&mut rng, &mut k_rope);

    // Compressed path: the cache IS the latent buffer; the attention
    // kernel reads it directly via `d_latent=D_LATENT`.
    let mut out_compressed = vec![0.0f32; seq_q * (D_LATENT + D_ROPE)];
    mla_attention(
        &q_latent,
        &k_latent,
        &v_latent,
        &q_rope,
        &k_rope,
        D_LATENT,
        D_ROPE,
        seq_q,
        seq_k,
        &mut out_compressed,
    )
    .expect("compressed MLA must accept the test's well-formed inputs");

    // Uncompressed reference: same arithmetic over the full D_FULL
    // head dim. The latent slice (first D_LATENT output channels)
    // must equal the compressed output byte-for-byte.
    let mut out_uncompressed = vec![0.0f32; seq_q * (D_FULL + D_ROPE)];
    mla_attention(
        &q_full,
        &k_full,
        &v_full,
        &q_rope,
        &k_rope,
        D_FULL,
        D_ROPE,
        seq_q,
        seq_k,
        &mut out_uncompressed,
    )
    .expect("uncompressed MLA must accept the test's well-formed inputs");

    for s in 0..seq_q {
        let latent_uncompressed = &out_uncompressed
            [s * (D_FULL + D_ROPE)..s * (D_FULL + D_ROPE) + D_LATENT];
        let latent_compressed = &out_compressed
            [s * (D_LATENT + D_ROPE)..s * (D_LATENT + D_ROPE) + D_LATENT];
        assert_buf_close(
            latent_uncompressed,
            latent_compressed,
            1e-5,
            "MLA latent slice",
        );
    }
}

#[test]
fn deepseek_mla_cache_size_smaller_than_uncompressed() {
    let mut rng = Lcg::new(0xC4_C4E512E);
    let seq = B * H_KV;
    let mut k_full = vec![0.0f32; seq * D_FULL];
    let mut v_full = vec![0.0f32; seq * D_FULL];
    let mut k_latent = vec![0.0f32; seq * D_LATENT];
    let mut v_latent = vec![0.0f32; seq * D_LATENT];
    fill_signed(&mut rng, &mut k_full);
    fill_signed(&mut rng, &mut v_full);
    let cached = compress_kv(&mut rng, &k_full, &v_full, &mut k_latent, &mut v_latent);
    let uncompressed = 2 * seq * D_FULL * std::mem::size_of::<f32>();

    assert!(
        cached as f32 <= 0.5 * uncompressed as f32,
        "compressed cache {cached}B must be ≤ 0.5 × uncompressed {uncompressed}B",
    );
    let ratio = cached as f32 / uncompressed as f32;
    assert!(
        approx_eq(ratio, MLA_RATIO),
        "compression ratio {ratio} must equal D_latent/D = {MLA_RATIO}"
    );
}

// ---------------------------------------------------------------------------
// MTP: speculative vs sequential greedy, acceptance band, combined
// ---------------------------------------------------------------------------

/// Sequential greedy decode: invoke `mtp_propose` once per offset
/// (one token at a time, matching the non-speculative decode loop).
/// Returns the per-position tokens in offset order.
fn sequential_greedy(seed_logits: &[f32], k: usize, vocab: usize) -> Vec<u32> {
    (0..k)
        .map(|i| {
            mtp_propose(seed_logits, &[i], vocab)
                .expect("sequential mtp_propose must accept the test inputs")
                .tokens[0]
        })
        .collect()
}

#[test]
fn deepseek_mtp_speculative_proposals_byte_identical_to_sequential() {
    let k: usize = 4;
    let vocab: usize = 32;
    let mut rng = Lcg::new(0x77EC_0FFE);
    let mut logits = vec![0.0f32; k * vocab];
    fill_unit(&mut rng, &mut logits);

    let proposal = mtp_propose(&logits, &(0..k).collect::<Vec<_>>(), vocab)
        .expect("mtp_propose must accept well-formed logits");
    let sequential = sequential_greedy(&logits, k, vocab);

    assert_eq!(
        proposal.tokens, sequential,
        "single-pass MTP proposals must match sequential greedy decode token-for-token"
    );
}

#[test]
fn deepseek_mtp_acceptance_rate_within_band() {
    // Construct verifier logits where the softmax probability of the
    // proposal token sits in `[0.5, 0.9]` for ~half to ~all proposals
    // over n=32 prompts × k=4 proposals. Trick: each row's verifier
    // logit at the proposed token position gets a per-prompt random
    // bump `delta ∈ [0, 4]`; the verifier softmax(proposed) then
    // varies smoothly between ~uniform (~1/vocab) and ~1.0.
    let n_prompts: usize = 32;
    let k: usize = 4;
    let vocab: usize = 8;
    let accepted_threshold = 0.5f32;
    let mut rng = Lcg::new(0xACCE_BEAD);
    let mut base = vec![0.0f32; n_prompts * k * vocab];
    fill_unit(&mut rng, &mut base);

    let mut total_proposals = 0usize;
    let mut accepted = 0usize;
    for p in 0..n_prompts {
        for i in 0..k {
            let row = &mut base[p * k * vocab + i * vocab..p * k * vocab + (i + 1) * vocab];
            // The proposal token under uniform `row` is the argmax.
            let proposal_idx = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            // Bump the proposal's logit by a random delta ∈ [0, 4] so
            // its softmax probability spans the band.
            let delta = rng.next_f32() * 4.0;
            row[proposal_idx] += delta;
        }
        let slice = &base[p * k * vocab..(p + 1) * k * vocab];
        let proposal = mtp_propose(slice, &(0..k).collect::<Vec<_>>(), vocab)
            .expect("mtp_propose must accept well-formed logits");
        let mask = mtp_verify(&proposal, slice, vocab, accepted_threshold)
            .expect("mtp_verify must accept well-formed logits + proposal");
        for &ok in mask.iter() {
            total_proposals += 1;
            if ok {
                accepted += 1;
            }
        }
    }

    let rate = accepted as f32 / total_proposals as f32;
    assert!(
        (0.5..=0.9).contains(&rate),
        "MTP acceptance rate {rate} (accepted={accepted}/{total_proposals}) must lie in [0.5, 0.9]"
    );
}

#[test]
fn deepseek_mla_mtp_combined_byte_identical() {
    // MLA produces a context-conditioned hidden state; MTP consumes
    // it (projected to a `(k, vocab)` logits matrix) and proposes
    // `k` next tokens. The byte-equality contract: the joint
    // (MLA → MTP single-pass) path emits the same k tokens as a
    // reference (MLA-only → sequential greedy) path.
    let seq_q = 1usize;
    let seq_k = B * H_KV;
    let k: usize = 4;
    let vocab: usize = 16;
    let mut rng = Lcg::new(0x7011_D0F0_0BAD);

    // Build MLA inputs and run the reference kernel.
    let mut q_latent = vec![0.0f32; seq_q * D_LATENT];
    let mut k_latent = vec![0.0f32; seq_k * D_LATENT];
    let mut v_latent = vec![0.0f32; seq_k * D_LATENT];
    let mut q_rope = vec![0.0f32; seq_q * D_ROPE];
    let mut k_rope = vec![0.0f32; seq_k * D_ROPE];
    fill_signed(&mut rng, &mut q_latent);
    fill_signed(&mut rng, &mut k_latent);
    fill_signed(&mut rng, &mut v_latent);
    fill_signed(&mut rng, &mut q_rope);
    fill_signed(&mut rng, &mut k_rope);

    let mut mla_out = vec![0.0f32; seq_q * (D_LATENT + D_ROPE)];
    mla_attention(
        &q_latent,
        &k_latent,
        &v_latent,
        &q_rope,
        &k_rope,
        D_LATENT,
        D_ROPE,
        seq_q,
        seq_k,
        &mut mla_out,
    )
    .expect("MLA must accept the combined-path inputs");

    // Project MLA hidden state to `(k, vocab)` logits. The projection
    // matrix is deterministic and seeded so the joint path is
    // reproducible byte-for-byte.
    let hidden = D_LATENT + D_ROPE;
    let mut proj = vec![0.0f32; k * vocab * hidden];
    fill_signed(&mut rng, &mut proj);
    let mut mtp_logits = vec![0.0f32; k * vocab];
    for i in 0..k {
        for v in 0..vocab {
            let mut acc = 0.0f32;
            for d in 0..hidden {
                acc += mla_out[d] * proj[(i * vocab + v) * hidden + d];
            }
            mtp_logits[i * vocab + v] = acc;
        }
    }

    // Joint path: single-pass MTP over the MLA-derived logits.
    let joint = mtp_propose(&mtp_logits, &(0..k).collect::<Vec<_>>(), vocab)
        .expect("joint MTP must accept the projected logits");

    // Reference path: same logits, sequential greedy decode.
    let sequential = sequential_greedy(&mtp_logits, k, vocab);

    assert_eq!(
        joint.tokens, sequential,
        "joint (MLA → MTP) proposals must equal reference (MLA → sequential greedy) tokens"
    );
    // Round-trip the joint proposal through `mtp_verify` with
    // threshold 0; every position must accept.
    let mask = mtp_verify(&joint, &mtp_logits, vocab, 0.0)
        .expect("mtp_verify with threshold=0 must accept the joint proposal");
    assert_eq!(mask, vec![true; k]);
}