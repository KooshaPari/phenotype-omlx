//! Temporal windowed attention parity test.

#![cfg(all(feature = "metal", target_os = "macos"))]

use metal_runtime::{temporal_window_attention_metal, ArtifactAllowlist, MetallibLoader};
use sha2::{Digest, Sha256};

#[test]
fn metal_matches_causal_window_reference() {
    let path =
        std::env::var("TEMPORAL_ATTN_METALLIB").expect("TEMPORAL_ATTN_METALLIB test artifact");
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
    let q = [1.0, 0.0, 0.5, 1.0, -1.0, 0.25, 0.0, 0.5];
    let k = [0.5, 1.0, 0.0, 1.0, 1.0, 0.25, -0.5, 0.5];
    let v = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -2.0];
    let tokens = 4;
    let dim = 2;
    let window = 2;
    let scale = 0.70710677;
    let actual =
        temporal_window_attention_metal(&q, &k, &v, tokens, 1, dim, window, scale, &artifact)
            .unwrap();
    let mut expected = vec![0.0; q.len()];
    for token in 0..tokens {
        let first = (token + 1).saturating_sub(window);
        let mut scores = Vec::new();
        for key in first..=token {
            scores.push(
                (0..dim)
                    .map(|d| q[token * dim + d] * k[key * dim + d])
                    .sum::<f32>()
                    * scale,
            );
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let denom: f32 = weights.iter().sum();
        for lane in 0..dim {
            expected[token * dim + lane] = weights
                .iter()
                .enumerate()
                .map(|(i, w)| w * v[(first + i) * dim + lane])
                .sum::<f32>()
                / denom;
        }
    }
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "mismatch: {actual} vs {expected}"
        );
    }
}
