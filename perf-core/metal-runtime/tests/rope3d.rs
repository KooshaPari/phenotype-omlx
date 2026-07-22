//! On-device 3D RoPE parity against a scalar reference.

#![cfg(all(feature = "metal", target_os = "macos"))]

use metal_runtime::{rope_3d_metal, ArtifactAllowlist, MetallibLoader};
use sha2::{Digest, Sha256};

fn reference(
    input: &[f32],
    positions: &[[u32; 4]],
    inv: &[f32],
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = input.to_vec();
    let pairs = head_dim / 6;
    for (token, pos) in positions.iter().enumerate() {
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            for lane in 0..(pairs * 6) {
                let pair = lane / 2;
                let axis_pair = pair % pairs;
                let axis = pair / pairs;
                let angle = pos[axis] as f32 * inv[axis_pair];
                let (s, c) = angle.sin_cos();
                let even = input[base + (lane & !1)];
                let odd = input[base + (lane | 1)];
                out[base + lane] = if lane & 1 == 0 {
                    even * c - odd * s
                } else {
                    even * s + odd * c
                };
            }
        }
    }
    out
}

#[test]
fn metal_matches_scalar_reference() {
    let path = std::env::var("ROPE_3D_METALLIB").expect("ROPE_3D_METALLIB test artifact");
    let bytes = std::fs::read(&path).unwrap();
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    let root = std::path::Path::new(&path).parent().unwrap();
    let name = std::path::Path::new(&path)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();
    let artifact = MetallibLoader::new(root, ArtifactAllowlist::new([(name.to_owned(), digest)]))
        .load(name)
        .unwrap();

    let positions = [[0, 1, 2, 0], [1, 2, 3, 0]];
    let inv = [0.125_f32];
    let q: Vec<f32> = (0..12).map(|i| i as f32 * 0.25 - 1.0).collect();
    let k: Vec<f32> = q.iter().map(|v| v * 0.5).collect();
    let (actual_q, actual_k) =
        rope_3d_metal(&q, &k, &positions, &inv, &inv, &inv, 1, 6, &artifact).unwrap();
    let expected_q = reference(&q, &positions, &inv, 1, 6);
    let expected_k = reference(&k, &positions, &inv, 1, 6);
    for (actual, expected) in actual_q.iter().zip(expected_q.iter()) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "q mismatch: {actual} vs {expected}"
        );
    }
    for (actual, expected) in actual_k.iter().zip(expected_k.iter()) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "k mismatch: {actual} vs {expected}"
        );
    }
}
