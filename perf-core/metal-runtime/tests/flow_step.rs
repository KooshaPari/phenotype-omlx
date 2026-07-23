//! Fused flow/CFG Metal parity test.

#![cfg(all(feature = "metal", target_os = "macos"))]

use metal_runtime::{flow_cfg_step_metal, ArtifactAllowlist, MetallibLoader};
use sha2::{Digest, Sha256};

#[test]
fn metal_matches_reference() {
    let path = std::env::var("FLOW_STEP_METALLIB").expect("FLOW_STEP_METALLIB test artifact");
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
    let x = [1.0, -2.0, 0.5, 3.0];
    let uncond = [0.5, 1.0, -1.0, 0.25];
    let cond = [1.5, 0.0, 2.0, -0.75];
    let scale = 1.5;
    let dt = 0.1;
    let actual = flow_cfg_step_metal(&x, &uncond, &cond, scale, dt, &artifact).unwrap();
    let expected: Vec<f32> = x
        .iter()
        .zip(uncond.iter().zip(cond.iter()))
        .map(|(x, (u, c))| x + dt * (u + scale * (c - u)))
        .collect();
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "mismatch: {actual} vs {expected}"
        );
    }
}
