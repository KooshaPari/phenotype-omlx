//! Metal parity for fused diffusion argmax/confidence.
#![cfg(all(feature = "metal", target_os = "macos"))]

use metal_runtime::{diffusion_argmax_confidence_metal, ArtifactAllowlist, MetallibLoader};
use sha2::{Digest, Sha256};

#[test]
fn metal_matches_stable_argmax_and_softmax_max() {
    let path = std::env::var("DIFFUSION_CONFIDENCE_METALLIB").expect("test artifact");
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
    let logits = [0.0f32, 2.0, 1.0, -1.0, 3.0, 3.0, 0.0, 0.0];
    let (ids, confidence) = diffusion_argmax_confidence_metal(&logits, 2, 4, &artifact).unwrap();
    assert_eq!(ids, [1, 0]);
    for row in 0..2 {
        let start = row * 4;
        let max = logits[start..start + 4]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let denom: f32 = logits[start..start + 4]
            .iter()
            .map(|v| (v - max).exp())
            .sum();
        assert!((confidence[row] - 1.0 / denom).abs() < 1e-5);
    }
}
