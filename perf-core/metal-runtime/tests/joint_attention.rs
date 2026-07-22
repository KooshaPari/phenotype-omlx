//! On-device joint-attention parity for Flux/SD3-shaped streams.

#![cfg(all(feature = "metal", target_os = "macos"))]

use metal_runtime::{joint_attention_metal, ArtifactAllowlist, MetallibLoader};
use sha2::{Digest, Sha256};

fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    qt: usize,
    kt: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; qt * d];
    for token in 0..qt {
        let qrow = &q[token * d..(token + 1) * d];
        let mut scores = Vec::with_capacity(kt);
        for key in 0..kt {
            let krow = &k[key * d..(key + 1) * d];
            scores.push(qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * scale);
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let denom: f32 = weights.iter().sum();
        for lane in 0..d {
            out[token * d + lane] = weights
                .iter()
                .enumerate()
                .map(|(key, weight)| weight * v[key * d + lane])
                .sum::<f32>()
                / denom;
        }
    }
    out
}

#[test]
fn metal_matches_scalar_reference() {
    let path =
        std::env::var("JOINT_ATTENTION_METALLIB").expect("JOINT_ATTENTION_METALLIB test artifact");
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
    let q = [0.5, -1.0, 1.5, 0.25];
    let k = [0.25, 0.5, -0.75, 1.0, 1.25, -0.5];
    let v = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5];
    let scale = 0.70710677;
    let actual = joint_attention_metal(&q, &k, &v, 2, 3, 1, 2, scale, &artifact).unwrap();
    let expected = reference(&q, &k, &v, 2, 3, 2, scale);
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "mismatch: {actual} vs {expected}"
        );
    }
}
