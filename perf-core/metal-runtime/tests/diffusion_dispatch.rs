//! Explicit opt-in device fixture for diffusion scheduler parity.
//!
//! This test is ignored by default and requires a caller-provided verified
//! artifact plus manifest. It is deliberately not a model or evaluation run.

#![cfg(all(feature = "metal", target_os = "macos"))]

use metal_runtime::{
    compare_f32, compare_u32, compare_u8, diffusion_active_compact_metal, diffusion_remask_metal,
    diffusion_trajectory_metal, ArtifactAllowlist, MetallibLoader,
};

#[test]
#[ignore = "explicit device fixture; requires METAL_RUNTIME_TEST_ARTIFACT and MANIFEST"]
fn diffusion_three_stage_fixture_matches_oracle() {
    let artifact_path = std::env::var("METAL_RUNTIME_TEST_ARTIFACT")
        .expect("set METAL_RUNTIME_TEST_ARTIFACT to an allowlisted metallib");
    let manifest_path = std::env::var("METAL_RUNTIME_TEST_MANIFEST")
        .expect("set METAL_RUNTIME_TEST_MANIFEST to its JSON allowlist");
    let manifest = std::fs::read(manifest_path).expect("read manifest");
    let root = std::path::Path::new(&artifact_path)
        .parent()
        .expect("artifact parent")
        .to_path_buf();
    let name = std::path::Path::new(&artifact_path)
        .file_name()
        .expect("artifact basename")
        .to_str()
        .expect("utf8 artifact name");
    let _ = ArtifactAllowlist::from_manifest_json(&manifest).expect("strict manifest");
    let artifact = MetallibLoader::from_manifest_json(root, &manifest)
        .expect("manifest loader")
        .load(name)
        .expect("allowlisted artifact");

    let values = [10_u32, 11, 12, 13, 14, 15, 16, 17];
    let active = [1_u8, 0, 1, 0, 1, 0, 0, 1];
    let (compacted, positions) =
        diffusion_active_compact_metal(&values, &active, &artifact).expect("compact dispatch");
    compare_u32("compacted values", &[10, 12, 14, 17], &compacted).unwrap();
    compare_u32("positions", &[0, 2, 4, 7], &positions).unwrap();

    let confidence = [0.9_f32, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.95];
    let next_mask =
        diffusion_remask_metal(&active, &confidence, 0.5, &artifact).expect("remask dispatch");
    compare_u8("next mask", &[1, 1, 1, 1, 1, 1, 1, 1], &next_mask).unwrap();

    let previous = [0.8_f32; 8];
    let entropy = [0.1_f32; 8];
    let (momentum, converged) =
        diffusion_trajectory_metal(&previous, &confidence, &entropy, 0.75, 0.15, &artifact)
            .expect("trajectory dispatch");
    compare_f32(
        "momentum",
        &[0.1, 0.6, 0.0, 0.7, 0.1, 0.5, 0.2, 0.15],
        &momentum,
        1e-5,
    )
    .unwrap();
    compare_u8("converged", &[1, 0, 1, 0, 0, 0, 0, 1], &converged).unwrap();
}
