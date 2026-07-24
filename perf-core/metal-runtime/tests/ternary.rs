//! On-device packed ternary GEMM parity for the Bonsai layout.

#![cfg(all(feature = "metal", target_os = "macos"))]

use metal_runtime::{ternary_gemm_metal, ArtifactAllowlist, MetallibLoader};
use sha2::{Digest, Sha256};

fn pack(values: &[u8], k: usize, n: usize) -> Vec<u8> {
    let stride = k.div_ceil(4);
    let mut packed = vec![0; n * stride];
    for col in 0..n {
        for d in 0..k {
            packed[col * stride + d / 4] |= values[col * k + d] << ((d & 3) * 2);
        }
    }
    packed
}

#[test]
fn metal_matches_scalar_reference() {
    let path = std::env::var("TERNARY_GEMM_METALLIB").expect("TERNARY_GEMM_METALLIB test artifact");
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
    let m = 2;
    let k = 5;
    let n = 3;
    let activations = [1.0, -2.0, 0.5, 3.0, -1.0, -0.5, 2.0, 1.5, 0.0, 4.0];
    let codes = [1, 2, 0, 1, 2, 2, 1, 1, 0, 1, 0, 1, 2, 2, 1];
    let scales = [0.5, 1.25, 2.0];
    let packed = pack(&codes, k, n);
    let actual = ternary_gemm_metal(&activations, &packed, &scales, m, k, n, &artifact).unwrap();
    let expected: Vec<f32> = (0..m)
        .flat_map(|row| {
            (0..n).map(move |col| {
                let sum: f32 = (0..k)
                    .map(|d| {
                        let weight = match codes[col * k + d] {
                            1 => 1.0,
                            2 => -1.0,
                            _ => 0.0,
                        };
                        activations[row * k + d] * weight
                    })
                    .sum();
                sum * scales[col]
            })
        })
        .collect();
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "mismatch: {actual} vs {expected}"
        );
    }
}

#[test]
fn metal_matches_edge_shape_with_all_packed_codes() {
    let path = std::env::var("TERNARY_GEMM_METALLIB").expect("TERNARY_GEMM_METALLIB test artifact");
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

    // K=127 exercises the partial final packed byte; code 3 is reserved and must decode as 0.
    let (m, k, n) = (1, 127, 256);
    let activations: Vec<f32> = (0..m * k)
        .map(|i| ((i * 17 % 101) as f32 - 50.0) / 13.0)
        .collect();
    let codes: Vec<u8> = (0..n * k).map(|i| (i % 4) as u8).collect();
    let scales: Vec<f32> = (0..n).map(|i| 0.25 + (i % 11) as f32 / 17.0).collect();
    let packed = pack(&codes, k, n);
    let actual = ternary_gemm_metal(&activations, &packed, &scales, m, k, n, &artifact).unwrap();
    for col in 0..n {
        let expected: f32 = (0..k)
            .map(|d| {
                let weight = match codes[col * k + d] {
                    1 => 1.0,
                    2 => -1.0,
                    _ => 0.0,
                };
                activations[d] * weight
            })
            .sum::<f32>()
            * scales[col];
        assert!(
            (actual[col] - expected).abs() < 1e-5,
            "col {col}: {} vs {expected}",
            actual[col]
        );
    }
}
