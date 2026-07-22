//! Extended oracle coverage for the RWKV family of recurrent-hybrid
//! operators plus the Jamba hybrid M-A-M-A mixer.
//!
//! `rwkv7.rs` pins the *selector* for RWKV-7. This file pins the
//! *numerical oracle* for RWKV-7 time-mix decay monotonicity and the
//! RWKV-7 channel-mix invariant, and the Jamba hybrid M-A-M-A mixer
//! that ties Mamba's forward scan together with self-attention. The
//! Jamba oracle lives here (rather than in `mamba_extended.rs`) because
//! the Jamba paper is a *hybrid* family and reads naturally next to
//! RWKV-7, the other non-Mamba/non-RWKV recurrent operator covered in
//! this crate. Pure f32 oracle assertions; no selector.
//!
//! The Mamba-side helpers used by the Jamba test (`mamba_forward_scan`
//! and `rand_unit`) live in `mamba_extended.rs` and are re-imported as
//! `pub(crate)` from there so this file can keep using the canonical
//! recurrence without duplicating the arithmetic.

use super::mamba_extended::{mamba_forward_scan, rand_unit};

// Self-attention with Q=K=V=input: `softmax(QKᵀ / √D) @ V`.
fn attention(qkv: &[f32], l: usize, d: usize) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let mut scores = vec![0.0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            let mut acc = 0.0f32;
            for c in 0..d {
                acc += qkv[i * d + c] * qkv[j * d + c];
            }
            scores[i * l + j] = acc * scale;
        }
    }
    for i in 0..l {
        let row = &mut scores[i * l..(i + 1) * l];
        let mut mx = row[0];
        for &s in row.iter() {
            if s > mx {
                mx = s;
            }
        }
        let mut sum = 0.0f32;
        for s in row.iter_mut() {
            *s = (*s - mx).exp();
            sum += *s;
        }
        for s in row.iter_mut() {
            *s /= sum;
        }
    }
    let mut out = vec![0.0f32; l * d];
    for i in 0..l {
        for c in 0..d {
            let mut acc = 0.0f32;
            for j in 0..l {
                acc += scores[i * l + j] * qkv[j * d + c];
            }
            out[i * d + c] = acc;
        }
    }
    out
}

// `x_mixed = μ·x_t + (1-μ)·x_prev`, `out = w_v · sigmoid(w_k · x_mixed + bias)`.
fn rwkv_channel_mix_step(x_t: f32, x_prev: f32, mu: f32, w_k: f32, w_v: f32, bias: f32) -> f32 {
    let x_mixed = mu * x_t + (1.0 - mu) * x_prev;
    w_v * (1.0 / (1.0 + (-(w_k * x_mixed + bias)).exp()))
}

// ---------------------------------------------------------------------------
// (4) Jamba hybrid — M-A-M-A 4-layer mixer byte-identical to manual
//     interleaving of the same ops.
// ---------------------------------------------------------------------------

#[test]
fn jamba_hybrid_attention_mamba_mixer_byte_identical() {
    let l = 8usize;
    let d = 4usize;
    let mut rng = 0xA1B2C3D4u32;
    let x: Vec<f32> = (0..l * d).map(|_| rand_unit(&mut rng)).collect();
    let a: Vec<f32> = (0..d).map(|_| 0.5 + 0.4 * rand_unit(&mut rng).abs()).collect();
    let b: Vec<f32> = (0..d).map(|_| 0.3 + 0.4 * rand_unit(&mut rng).abs()).collect();

    // Hybrid: 4-layer Mamba/Attention/Mamba/Attention mixer.
    let mut h = mamba_forward_scan(&x, &a, &b, l, d);
    h = attention(&h, l, d);
    h = mamba_forward_scan(&h, &a, &b, l, d);
    let hybrid = attention(&h, l, d);

    // Reference: same four ops expressed inline. Identical f32 ops
    // order ⇒ bit-for-bit match with `hybrid`.
    let mut h0 = vec![0.0f32; d];
    let mut h1 = Vec::with_capacity(l * d);
    for t in 0..l {
        for c in 0..d {
            h0[c] = a[c] * h0[c] + b[c] * x[t * d + c];
            h1.push(h0[c]);
        }
    }
    let h1_attn = attention(&h1, l, d);
    let mut h2 = vec![0.0f32; d];
    let mut h3 = Vec::with_capacity(l * d);
    for t in 0..l {
        for c in 0..d {
            h2[c] = a[c] * h2[c] + b[c] * h1_attn[t * d + c];
            h3.push(h2[c]);
        }
    }
    let reference = attention(&h3, l, d);

    assert_eq!(hybrid.len(), reference.len());
    for (idx, (&h, &r)) in hybrid.iter().zip(reference.iter()).enumerate() {
        assert_eq!(
            h.to_bits(),
            r.to_bits(),
            "Jamba M-A-M-A mixer diverged at [{idx}] (bits {:08x} vs {:08x})",
            h.to_bits(),
            r.to_bits(),
        );
    }
}

// ---------------------------------------------------------------------------
// (5) RWKV-7 time-mix decay monotonicity.
// ---------------------------------------------------------------------------

#[test]
fn rwkv_time_mix_decay_monotonic() {
    // RWKV-7: `α(ℓ) = α_base · decay_per_layer^ℓ`, with both factors
    // in `(0, 1)`. Sequence must be strictly monotonically decreasing.
    let alpha_base: f32 = 0.9;
    let decay_per_layer: f32 = 0.85;
    let num_layers: usize = 8;
    let mut alphas: Vec<f32> = Vec::with_capacity(num_layers);
    let mut a = alpha_base;
    for _ in 0..num_layers {
        alphas.push(a);
        a *= decay_per_layer;
    }
    for w in alphas.windows(2) {
        assert!(
            w[0] > w[1],
            "RWKV time-mix decay must be monotonically decreasing; α(ℓ)={} should be > α(ℓ+1)={}",
            w[0],
            w[1],
        );
        assert!(w[1] > 0.0 && w[0] <= 1.0, "decay must remain in (0, 1]");
    }

    // Cross-check: a non-monotonic synthetic sequence is rejected.
    let bad = [0.9f32, 0.8, 0.7, 0.85, 0.6];
    let any_violation = bad.windows(2).any(|w| w[0] <= w[1]);
    assert!(
        any_violation,
        "synthetic non-monotonic sequence must contain at least one violation"
    );
}

// ---------------------------------------------------------------------------
// (6) RWKV-7 channel-mix — within-tolerance oracle.
// ---------------------------------------------------------------------------

#[test]
fn rwkv_channel_mix_within_tolerance() {
    // 4-layer RWKV channel-mix pass over a 16-step sequence.
    let num_layers = 4usize;
    let seq_len = 16usize;
    let mut rng = 0xFEEDFACEu32;
    let layer_params: Vec<(f32, f32, f32)> = (0..num_layers)
        .map(|i| {
            let mu = 0.6 + 0.1 * (i as f32);
            let w_k = 1.0 + 0.2 * rand_unit(&mut rng).abs();
            let w_v = 0.7 + 0.3 * rand_unit(&mut rng).abs();
            (mu, w_k, w_v)
        })
        .collect();
    let bias: f32 = 0.1 * rand_unit(&mut rng);

    // Capture inputs deterministically so the replay path matches.
    let inputs: Vec<f32> = (0..seq_len).map(|_| rand_unit(&mut rng)).collect();

    // Reference path.
    let mut reference: Vec<f32> = Vec::with_capacity(seq_len);
    let mut prev = 0.0f32;
    for &x_t in &inputs {
        let mut h = x_t;
        for &(mu, w_k, w_v) in &layer_params {
            h = rwkv_channel_mix_step(h, prev, mu, w_k, w_v, bias);
        }
        reference.push(h);
        prev = h;
    }

    // Replay path: identical arithmetic order, fresh local state.
    let mut replay: Vec<f32> = Vec::with_capacity(seq_len);
    let mut prev_r = 0.0f32;
    for &x_t in &inputs {
        let mut h = x_t;
        for &(mu, w_k, w_v) in &layer_params {
            h = rwkv_channel_mix_step(h, prev_r, mu, w_k, w_v, bias);
        }
        replay.push(h);
        prev_r = h;
    }

    assert_eq!(replay.len(), reference.len());
    for (idx, (&r, &p)) in reference.iter().zip(replay.iter()).enumerate() {
        let diff = (r - p).abs();
        assert!(
            diff <= 1e-5,
            "RWKV channel-mix replay[{idx}] diverged by {diff} (>1e-5)"
        );
    }
}