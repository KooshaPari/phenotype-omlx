//! Production-shape Mamba scan percentile probe; intentionally separate from smoke baselines.
#![cfg(all(feature = "metal", target_os = "macos"))]

use metal_runtime::{
    mamba_selective_scan_metal, mamba_selective_step_metal, ArtifactAllowlist, MetallibLoader,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::time::Instant;

#[derive(Serialize)]
struct Record {
    schema: &'static str,
    workload: &'static str,
    steps: usize,
    state_dim: usize,
    samples: usize,
    fused_median_us: f64,
    fused_p95_us: f64,
    repeated_step_median_us: f64,
    repeated_step_p95_us: f64,
    fused_artifact_sha256: String,
    step_artifact_sha256: String,
}

#[test]
#[ignore = "explicit production-shape artifact-backed benchmark"]
fn production_shape_mamba_scan_percentiles() {
    let path = std::path::PathBuf::from(std::env::var("MAMBA_SCAN_METALLIB").unwrap());
    let step_path = std::path::PathBuf::from(std::env::var("MAMBA_METALLIB").unwrap());
    let bytes = std::fs::read(&path).unwrap();
    let step_bytes = std::fs::read(&step_path).unwrap();
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    let sha = digest.iter().map(|b| format!("{b:02x}")).collect();
    let step_sha = Sha256::digest(step_bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let name = path.file_name().unwrap().to_str().unwrap().to_owned();
    let artifact = MetallibLoader::new(
        path.parent().unwrap(),
        ArtifactAllowlist::new([(name.clone(), digest)]),
    )
    .load(&name)
    .unwrap();
    let step_name = step_path.file_name().unwrap().to_str().unwrap().to_owned();
    let step_artifact = MetallibLoader::new(
        step_path.parent().unwrap(),
        ArtifactAllowlist::new([(
            step_name.clone(),
            Sha256::digest(std::fs::read(&step_path).unwrap()).into(),
        )]),
    )
    .load(&step_name)
    .unwrap();
    let steps = 256;
    let state_dim = 64;
    let u: Vec<f32> = (0..steps).map(|i| (i as f32 * 0.013).sin()).collect();
    let dt: Vec<f32> = (0..steps).map(|i| 0.05 + (i % 7) as f32 * 0.005).collect();
    let b: Vec<f32> = (0..steps).map(|i| 0.2 + (i % 5) as f32 * 0.03).collect();
    let c: Vec<f32> = (0..steps).map(|i| (i as f32 * 0.021).cos()).collect();
    let d: Vec<f32> = (0..steps).map(|i| (i % 3) as f32 * 0.01).collect();
    let a_log: Vec<f32> = (0..state_dim).map(|i| -0.1 - i as f32 * 0.01).collect();
    let mut fused_samples = Vec::new();
    let mut repeated_samples = Vec::new();
    for _ in 0..9 {
        let mut state = vec![0.0; state_dim];
        let start = Instant::now();
        mamba_selective_scan_metal(&u, &dt, &b, &c, &d, &a_log, &mut state, &artifact).unwrap();
        fused_samples.push(start.elapsed().as_secs_f64() * 1e6);
        let mut step_state = vec![0.0; state_dim];
        let start = Instant::now();
        for t in 0..steps {
            mamba_selective_step_metal(
                u[t],
                dt[t],
                b[t],
                c[t],
                d[t],
                &a_log,
                &mut step_state,
                &step_artifact,
            )
            .unwrap();
        }
        repeated_samples.push(start.elapsed().as_secs_f64() * 1e6);
    }
    fused_samples.sort_by(f64::total_cmp);
    repeated_samples.sort_by(f64::total_cmp);
    let record = Record {
        schema: "metal-runtime.mamba-production-percentiles.v1",
        workload: "production-shape",
        steps,
        state_dim,
        samples: fused_samples.len(),
        fused_median_us: fused_samples[fused_samples.len() / 2],
        fused_p95_us: fused_samples[(fused_samples.len() - 1) * 95 / 100],
        repeated_step_median_us: repeated_samples[repeated_samples.len() / 2],
        repeated_step_p95_us: repeated_samples[(repeated_samples.len() - 1) * 95 / 100],
        fused_artifact_sha256: sha,
        step_artifact_sha256: step_sha,
    };
    if let Ok(path) = std::env::var("MAMBA_PRODUCTION_BENCH_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    }
}
