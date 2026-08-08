//! Explicit opt-in device fixture for diffusion scheduler parity.
//!
//! This test is ignored by default and requires a caller-provided verified
//! artifact plus manifest. It is deliberately not a model or evaluation run.

#![cfg(all(feature = "metal", target_os = "macos"))]

use metal_runtime::{
    compare_f32, compare_u32, compare_u8, diffusion_active_compact_metal_with_telemetry,
    diffusion_remask_metal_with_telemetry, diffusion_trajectory_metal_with_telemetry,
    ArtifactAllowlist, DiffusionDispatchPlan, DiffusionDispatchTelemetry, MetallibLoader,
};

#[test]
fn compaction_oracle_canonicalizes_atomic_output_order() {
    let mut compacted = vec![14_u32, 10, 17, 12];
    let mut positions = vec![4_u32, 0, 7, 2];

    canonicalize_compacted_pairs(&mut compacted, &mut positions);

    assert_eq!(compacted, vec![10, 12, 14, 17]);
    assert_eq!(positions, vec![0, 2, 4, 7]);
}

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
    let compacted_outcome =
        diffusion_active_compact_metal_with_telemetry(&values, &active, &artifact)
            .expect("compact telemetry");
    assert!(compacted_outcome.telemetry.completed);
    assert!(!compacted_outcome.telemetry.fallback);
    let (mut compacted, mut positions) = compacted_outcome.output.expect("compact dispatch");
    canonicalize_compacted_pairs(&mut compacted, &mut positions);
    compare_u32("compacted values", &[10, 12, 14, 17], &compacted).unwrap();
    compare_u32("positions", &[0, 2, 4, 7], &positions).unwrap();

    let confidence = [0.9_f32, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.95];
    let remask_outcome =
        diffusion_remask_metal_with_telemetry(&active, &confidence, 0.5, &artifact)
            .expect("remask telemetry");
    assert!(remask_outcome.telemetry.completed);
    assert!(!remask_outcome.telemetry.fallback);
    let next_mask = remask_outcome.output.expect("remask dispatch");
    // Token 6 was already accepted and remains above the confidence floor;
    // every other token is either still a candidate or below the floor.
    compare_u8("next mask", &[1, 1, 1, 1, 1, 1, 0, 1], &next_mask).unwrap();

    let previous = [0.8_f32; 8];
    let entropy = [0.1_f32; 8];
    let trajectory_outcome = diffusion_trajectory_metal_with_telemetry(
        &previous,
        &confidence,
        &entropy,
        0.75,
        0.15,
        &artifact,
    )
    .expect("trajectory telemetry");
    assert!(trajectory_outcome.telemetry.completed);
    assert!(!trajectory_outcome.telemetry.fallback);
    let (momentum, converged) = trajectory_outcome.output.expect("trajectory dispatch");
    let plan = DiffusionDispatchPlan::for_tokens(values.len()).expect("dispatch plan");
    let report = DiffusionDispatchTelemetry::for_plan(
        &plan,
        [
            compacted_outcome.telemetry,
            remask_outcome.telemetry,
            trajectory_outcome.telemetry,
        ],
    )
    .expect("dispatch report");
    assert!(report.all_completed());
    assert!(!report.used_fallback());
    compare_f32(
        "momentum",
        &[0.1, 0.6, 0.0, 0.7, 0.1, 0.5, 0.2, 0.15],
        &momentum,
        1e-5,
    )
    .unwrap();
    compare_u8("converged", &[1, 0, 1, 0, 0, 0, 0, 1], &converged).unwrap();
}

fn canonicalize_compacted_pairs(values: &mut [u32], positions: &mut [u32]) {
    assert_eq!(
        values.len(),
        positions.len(),
        "compaction pair lengths must match"
    );
    let mut pairs: Vec<_> = positions
        .iter()
        .copied()
        .zip(values.iter().copied())
        .collect();
    pairs.sort_unstable_by_key(|(position, _)| *position);
    for ((position, value), (out_position, out_value)) in pairs
        .into_iter()
        .zip(positions.iter_mut().zip(values.iter_mut()))
    {
        *out_position = position;
        *out_value = value;
    }
}
