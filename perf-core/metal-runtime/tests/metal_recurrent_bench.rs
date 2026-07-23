//! Percentile baseline for recurrent and compressed-context Metal bindings.
#![cfg(all(feature = "metal", target_os = "macos"))]

use metal_runtime::{
    cca_block_attend_metal, deltanet_step_metal, mamba_selective_scan_metal,
    mamba_selective_step_metal, mla_cache_attend_metal, retnet_retention_step_metal,
    ArtifactAllowlist, MetallibLoader,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::time::Instant;

const SAMPLES: usize = 9;

#[derive(Serialize)]
struct Measurement {
    median_us: f64,
    p95_us: f64,
}

#[derive(Serialize)]
struct Record {
    schema: &'static str,
    samples: usize,
    measurements: std::collections::BTreeMap<String, Measurement>,
    artifacts: std::collections::BTreeMap<String, String>,
}

fn artifact(var: &str) -> (metal_runtime::MetallibArtifact, String) {
    let path = std::path::PathBuf::from(std::env::var(var).unwrap());
    let bytes = std::fs::read(&path).unwrap();
    let digest = Sha256::digest(&bytes);
    let hex = digest.iter().map(|b| format!("{b:02x}")).collect();
    let name = path.file_name().unwrap().to_str().unwrap().to_owned();
    let allow = ArtifactAllowlist::new([(name.clone(), digest.into())]);
    (
        MetallibLoader::new(path.parent().unwrap(), allow)
            .load(&name)
            .unwrap(),
        hex,
    )
}

fn percentile(mut values: Vec<f64>) -> Measurement {
    values.sort_by(f64::total_cmp);
    Measurement {
        median_us: values[values.len() / 2],
        p95_us: values[(values.len() - 1) * 95 / 100],
    }
}

#[test]
#[ignore = "explicit artifact-backed recurrent baseline"]
fn recurrent_bindings_percentiles() {
    let (delta, delta_hash) = artifact("DELTANET_METALLIB");
    let (cca, cca_hash) = artifact("CCA_METALLIB");
    let (mla, mla_hash) = artifact("MLA_CACHE_METALLIB");
    let (retnet, retnet_hash) = artifact("RETNET_METALLIB");
    let (mamba, mamba_hash) = artifact("MAMBA_METALLIB");
    let (mamba_scan, mamba_scan_hash) = artifact("MAMBA_SCAN_METALLIB");
    let q = [1.0_f32, 2.0, -1.0, 0.5, -0.25, 0.75, 1.5, -0.5];
    let k = [0.5_f32, -0.25, 1.0, 0.75, 0.1, -0.5, 0.25, 0.9];
    let v = [1.0_f32, -0.5, 0.25, 0.2, 0.75, -0.1, 0.3, -0.8];
    let summaries: Vec<f32> = (0..32).map(|i| (i as f32 * 0.17).sin()).collect();
    let scales = [0.8_f32, 1.1, 0.6, 1.3];
    let sizes = [8_u32, 8, 8, 8];
    let mla_q = [0.2_f32, -0.4, 0.6, 0.8];
    let mla_qr = [0.3_f32, -0.1];
    let mla_kv: Vec<f32> = (0..16).map(|i| (i as f32 * 0.11).cos()).collect();
    let mla_kr: Vec<f32> = (0..8).map(|i| (i as f32 * 0.07).sin()).collect();
    let mut delta_samples = Vec::with_capacity(SAMPLES);
    let mut cca_samples = Vec::with_capacity(SAMPLES);
    let mut mla_samples = Vec::with_capacity(SAMPLES);
    let mut retnet_samples = Vec::with_capacity(SAMPLES);
    let mut mamba_samples = Vec::with_capacity(SAMPLES);
    let mut mamba_scan_samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let mut state = vec![0.1_f32; 64];
        let start = Instant::now();
        deltanet_step_metal(&q, &k, &v, &mut state, 0.5, 8, &delta).unwrap();
        delta_samples.push(start.elapsed().as_secs_f64() * 1e6);
        let start = Instant::now();
        cca_block_attend_metal(&q, &summaries, &scales, &sizes, 8, &cca).unwrap();
        cca_samples.push(start.elapsed().as_secs_f64() * 1e6);
        let start = Instant::now();
        mla_cache_attend_metal(&mla_q, &mla_qr, &mla_kv, &mla_kr, 4, 2, &mla).unwrap();
        mla_samples.push(start.elapsed().as_secs_f64() * 1e6);
        let mut retention_state = vec![0.1_f32; 64];
        let start = Instant::now();
        retnet_retention_step_metal(&q, &k, &v, &mut retention_state, 0.8, 8, &retnet).unwrap();
        retnet_samples.push(start.elapsed().as_secs_f64() * 1e6);
        let mut mamba_state = vec![0.1_f32; 8];
        let a_log = [-0.1_f32, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8];
        let start = Instant::now();
        mamba_selective_step_metal(0.75, 0.1, 0.5, 0.25, 0.05, &a_log, &mut mamba_state, &mamba)
            .unwrap();
        mamba_samples.push(start.elapsed().as_secs_f64() * 1e6);
        let mut scan_state = vec![0.1_f32; 8];
        let scan_u = [0.7_f32, -0.2, 0.4, 1.1, -0.3, 0.2, 0.8, -0.6];
        let scan_dt = [0.4_f32, 0.2, 0.3, 0.1, 0.5, 0.2, 0.4, 0.3];
        let scan_b = [0.8_f32, 0.5, 0.7, 0.2, 0.4, 0.6, 0.3, 0.5];
        let scan_c = [-0.6_f32, -0.2, 0.4, 0.3, 0.1, -0.3, 0.2, 0.5];
        let scan_d = [0.2_f32, 0.1, 0.0, -0.1, 0.05, 0.0, 0.1, -0.05];
        let scan_a = [-0.1_f32, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8];
        let start = Instant::now();
        mamba_selective_scan_metal(
            &scan_u,
            &scan_dt,
            &scan_b,
            &scan_c,
            &scan_d,
            &scan_a,
            &mut scan_state,
            &mamba_scan,
        )
        .unwrap();
        mamba_scan_samples.push(start.elapsed().as_secs_f64() * 1e6);
    }
    let measurements = [
        ("deltanet_step_f32", percentile(delta_samples)),
        ("cca_block_attend_f32", percentile(cca_samples)),
        ("mla_cache_attend_f32", percentile(mla_samples)),
        ("retnet_retention_step_f32", percentile(retnet_samples)),
        ("mamba_selective_step_f32", percentile(mamba_samples)),
        ("mamba_selective_scan_f32", percentile(mamba_scan_samples)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect();
    let artifacts = [
        ("DELTANET_METALLIB", delta_hash),
        ("CCA_METALLIB", cca_hash),
        ("MLA_CACHE_METALLIB", mla_hash),
        ("RETNET_METALLIB", retnet_hash),
        ("MAMBA_METALLIB", mamba_hash),
        ("MAMBA_SCAN_METALLIB", mamba_scan_hash),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect();
    let record = Record {
        schema: "metal-runtime.recurrent-percentiles.v1",
        samples: SAMPLES,
        measurements,
        artifacts,
    };
    if let Ok(path) = std::env::var("METAL_RECURRENT_BENCH_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    }
}
