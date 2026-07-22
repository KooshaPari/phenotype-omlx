//! AdaLN/RMSNorm Metal parity test.

#![cfg(all(feature = "metal", target_os = "macos"))]

use metal_runtime::{adaln_rms_metal, ArtifactAllowlist, MetallibLoader};
use sha2::{Digest, Sha256};

#[test]
fn metal_matches_reference() {
    let path = std::env::var("ADALN_METALLIB").expect("ADALN_METALLIB test artifact");
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
    let x = [1.0, -2.0, 0.5, 3.0, -1.5, 2.5];
    let scale = [0.1, -0.2, 0.0, 0.3, 0.2, -0.1];
    let shift = [0.5, 0.0, -0.25, 0.1, -0.2, 0.4];
    let epsilon = 1e-5;
    let actual = adaln_rms_metal(&x, &scale, &shift, 2, 3, epsilon, &artifact).unwrap();
    let mut expected = Vec::new();
    for row in 0..2 {
        let values = &x[row * 3..row * 3 + 3];
        let inv = (values.iter().map(|v| v * v).sum::<f32>() / 3.0 + epsilon)
            .sqrt()
            .recip();
        for lane in 0..3 {
            let i = row * 3 + lane;
            expected.push(values[lane] * inv * (1.0 + scale[i]) + shift[i]);
        }
    }
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "mismatch: {actual} vs {expected}"
        );
    }
}
