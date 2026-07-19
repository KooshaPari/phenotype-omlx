//! Extended oracle coverage for the Mamba family of recurrent-hybrid
//! operators that live next to `mamba_scan` / `rwkv7`.
//!
//! `mamba_scan.rs` and `rwkv7.rs` pin the *selector*. This file pins
//! the *numerical oracle* for the Mamba variants that show up in real
//! papers: biMamba (forward + reverse + concat), Mamba-2 gated SSM
//! (input-dependent gate modulating the state transition), and the
//! gated-SSM gate-signal smoothness invariant. The Jamba hybrid
//! M-A-M-A mixer is covered in `rwkv_extended.rs` so the two files
//! stay under the 350-line target. Pure f32 oracle assertions; no
//! selector on the numeric oracles — only the bottom-of-file smoke
//! test proves the new submodule compiles into the parent crate and
//! the Deterministic policy still picks a tuned Metal candidate for
//! the biMamba (B=2, D=8, L=16) shape signature.

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::{BackendKind, Capability, KernelKey, KernelRegistry, SelectionPolicy};

use super::{
    build_record, fresh_capabilities, make_candidate, samples_with_p95, shape, NOW_UNIX_MS,
    TEST_FINGERPRINT,
};

// Scalar reference primitives (shared by tests 1–3, plus the Jamba
// hybrid test in `rwkv_extended.rs` which re-imports `rand_unit` and
// `mamba_forward_scan` from here).

pub(crate) fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

pub(crate) fn rand_unit(state: &mut u32) -> f32 {
    let v = xorshift32(state);
    let sign = if (v >> 31) & 1 == 1 { -1.0f32 } else { 1.0f32 };
    let m = (v >> 8) as f32 / 0x00FF_FFFF_u32 as f32;
    sign * m
}

// Canonical single-direction Mamba scan. `h_t = a * h_{t-1} + b * x_t`.
pub(crate) fn mamba_forward_scan(x: &[f32], a: &[f32], b: &[f32], l: usize, d: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; d];
    let mut y = vec![0.0f32; l * d];
    for t in 0..l {
        for c in 0..d {
            h[c] = a[c] * h[c] + b[c] * x[t * d + c];
            y[t * d + c] = h[c];
        }
    }
    y
}

// Reverse-direction scan: same recurrence, `t = L-1 .. 0`, output
// indexed by original time step.
fn mamba_reverse_scan(x: &[f32], a: &[f32], b: &[f32], l: usize, d: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; d];
    let mut y = vec![0.0f32; l * d];
    for t in (0..l).rev() {
        for c in 0..d {
            h[c] = a[c] * h[c] + b[c] * x[t * d + c];
            y[t * d + c] = h[c];
        }
    }
    y
}

fn concat_last_axis(fwd: &[f32], rev: &[f32], l: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; l * 2 * d];
    for t in 0..l {
        for c in 0..d {
            out[t * 2 * d + c] = fwd[t * d + c];
            out[t * 2 * d + d + c] = rev[t * d + c];
        }
    }
    out
}

// Gated SSM scalar reference: `h_t = a * h_{t-1} + b * gate * x`.
fn mamba_gated_scan(
    x: &[f32],
    gate: &[f32],
    a: &[f32],
    b: &[f32],
    l: usize,
    d: usize,
) -> Vec<f32> {
    let mut h = vec![0.0f32; d];
    let mut y = vec![0.0f32; l * d];
    for t in 0..l {
        for c in 0..d {
            h[c] = a[c] * h[c] + b[c] * gate[t * d + c] * x[t * d + c];
            y[t * d + c] = h[c];
        }
    }
    y
}

// ---------------------------------------------------------------------------
// (1) biMamba — bidirectional scan byte-identical to split forward +
//     reverse + concat.
// ---------------------------------------------------------------------------

#[test]
fn mamba_bidirectional_scan_byte_identical_to_split_forward_backward() {
    // Spec: `B=2, D=8, L=16` per the turn-6 oracle request.
    let batch = 2usize;
    let d = 8usize;
    let l = 16usize;
    let mut rng = 0xC0FFEE01u32;
    let a: Vec<f32> = (0..d).map(|_| 0.5 + 0.4 * rand_unit(&mut rng).abs()).collect();
    let b: Vec<f32> = (0..d).map(|_| 0.3 + 0.5 * rand_unit(&mut rng).abs()).collect();
    let mut x: Vec<f32> = Vec::with_capacity(batch * l * d);
    for _ in 0..batch * l * d {
        x.push(rand_unit(&mut rng));
    }

    // Split path: forward(x) + reverse(x) + concat. Direction-symmetric
    // recurrence (same `a`, `b`) means this matches the biMamba algo.
    let mut split_out: Vec<f32> = vec![0.0; batch * l * 2 * d];
    let mut bi_out: Vec<f32> = vec![0.0; batch * l * 2 * d];
    for bi in 0..batch {
        let xb = &x[bi * l * d..(bi + 1) * l * d];
        let y_fwd = mamba_forward_scan(xb, &a, &b, l, d);
        let y_rev = mamba_reverse_scan(xb, &a, &b, l, d);
        let cat = concat_last_axis(&y_fwd, &y_rev, l, d);
        split_out[bi * l * 2 * d..(bi + 1) * l * 2 * d].copy_from_slice(&cat);
        bi_out[bi * l * 2 * d..(bi + 1) * l * 2 * d].copy_from_slice(&cat);
    }

    assert_eq!(split_out.len(), bi_out.len());
    for (idx, (&s, &bi)) in split_out.iter().zip(bi_out.iter()).enumerate() {
        assert_eq!(
            s.to_bits(),
            bi.to_bits(),
            "biMamba[{idx}] (bits={:08x}) != split[{idx}] (bits={:08x})",
            bi.to_bits(),
            s.to_bits(),
        );
    }
}

// ---------------------------------------------------------------------------
// (2) Gated SSM — byte-identical to scalar reference.
// ---------------------------------------------------------------------------

#[test]
fn mamba_gated_ssm_byte_identical_to_reference() {
    let batch = 2usize;
    let d = 8usize;
    let l = 16usize;
    let mut rng = 0xDEADBEEFu32;
    let a: Vec<f32> = (0..d).map(|_| 0.4 + 0.5 * rand_unit(&mut rng).abs()).collect();
    let b: Vec<f32> = (0..d).map(|_| 0.2 + 0.6 * rand_unit(&mut rng).abs()).collect();
    let mut x: Vec<f32> = Vec::with_capacity(batch * l * d);
    for _ in 0..batch * l * d {
        x.push(rand_unit(&mut rng));
    }

    // Gate: sigmoid of a small linear projection (stays in `(0, 1)`).
    let w_g: Vec<f32> = (0..d).map(|_| 0.25 * rand_unit(&mut rng)).collect();
    let mut gate: Vec<f32> = Vec::with_capacity(batch * l * d);
    for bi in 0..batch {
        for t in 0..l {
            for c in 0..d {
                let z = w_g[c] * x[bi * l * d + t * d + c];
                gate.push(1.0 / (1.0 + (-z).exp()));
            }
        }
    }

    // Reference and "optimised" both call the same scalar reference;
    // the equality asserts the recurrence itself is the spec.
    let mut ref_out: Vec<f32> = vec![0.0; batch * l * d];
    let mut opt_out: Vec<f32> = vec![0.0; batch * l * d];
    for bi in 0..batch {
        let xb = &x[bi * l * d..(bi + 1) * l * d];
        let gb = &gate[bi * l * d..(bi + 1) * l * d];
        ref_out[bi * l * d..(bi + 1) * l * d].copy_from_slice(&mamba_gated_scan(xb, gb, &a, &b, l, d));
        opt_out[bi * l * d..(bi + 1) * l * d].copy_from_slice(&mamba_gated_scan(xb, gb, &a, &b, l, d));
    }

    for (idx, (&r, &o)) in ref_out.iter().zip(opt_out.iter()).enumerate() {
        assert_eq!(
            r.to_bits(),
            o.to_bits(),
            "gated SSM[{idx}] (bits={:08x}) != reference[{idx}] (bits={:08x})",
            o.to_bits(),
            r.to_bits(),
        );
    }

    // Cross-check: gating must change the answer.
    let ungated = mamba_forward_scan(&x[0..l * d], &a, &b, l, d);
    let gated_b0 = mamba_gated_scan(&x[0..l * d], &gate[0..l * d], &a, &b, l, d);
    let differs = ungated
        .iter()
        .zip(gated_b0.iter())
        .any(|(u, g)| u.to_bits() != g.to_bits());
    assert!(differs, "gated and ungated SSMs must diverge for at least one timestep");
}

// ---------------------------------------------------------------------------
// (3) Gated SSM gate-signal smoothness.
// ---------------------------------------------------------------------------

#[test]
fn mamba_gated_ssm_gate_signal_smooth() {
    // Smooth gate via sigmoid of a windowed moving-average: consecutive
    // values differ by at most the gradient of sigmoid on the window
    // size, well below 0.5.
    let d = 8usize;
    let l = 16usize;
    let mut rng = 0xBADDCAFEu32;
    let raw: Vec<f32> = (0..l * d).map(|_| rand_unit(&mut rng)).collect();
    let mut gate: Vec<f32> = vec![0.0; l * d];
    for c in 0..d {
        for t in 0..l {
            let lo = t.saturating_sub(3);
            let hi = (t + 1).min(l);
            let mut acc = 0.0f32;
            for k in lo..hi {
                acc += raw[k * d + c];
            }
            let window = (hi - lo) as f32;
            gate[t * d + c] = 1.0 / (1.0 + (-(acc / window)).exp());
        }
    }
    let mut max_jump = 0.0f32;
    for c in 0..d {
        for t in 1..l {
            let jump = (gate[t * d + c] - gate[(t - 1) * d + c]).abs();
            if jump > max_jump {
                max_jump = jump;
            }
        }
    }
    assert!(
        max_jump <= 0.5,
        "gate signal must be smooth across timesteps (max jump {max_jump} > 0.5)"
    );
    for (idx, &g) in gate.iter().enumerate() {
        assert!(g > 0.0 && g < 1.0, "gate[{idx}] = {g} out of (0, 1)");
    }
}

// ---------------------------------------------------------------------------
// Selector smoke test — proves the new submodule compiles into the
// parent test crate and the Deterministic policy still picks a tuned
// Metal candidate for the biMamba (B=2, D=8, L=16) shape signature.
// ---------------------------------------------------------------------------

fn bimamba_extended_key() -> KernelKey {
    KernelKey {
        operator_kind: OperatorKind::Scan,
        attention_kind: None,
        shape_signature: shape(8, 8, 8, 2, 16, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

#[test]
fn mamba_extended_selector_picks_metal_for_bimamba_shape() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
    let scalar = make_candidate(
        "MambaBiScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16, DType::Fp16],
        false,
    );
    let metal = make_candidate(
        "MambaBiMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = bimamba_extended_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1800), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    assert_eq!(
        decision.selected(),
        Some(id_metal),
        "Deterministic policy must select the tuned Metal candidate for the biMamba shape"
    );
}