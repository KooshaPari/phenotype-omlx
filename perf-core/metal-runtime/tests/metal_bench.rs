//! Artifact-backed percentile baseline for the public Metal kernel bindings.
//!
//! Run with `--ignored` and the six `*_METALLIB` variables set. Samples include
//! binding validation, pipeline creation, command submission, and synchronization;
//! this is intentionally an end-to-end baseline, not a pure GPU timestamp.

#![cfg(all(feature = "metal", target_os = "macos"))]

use metal_runtime::{
    adaln_rms_metal, diffusion_argmax_confidence_metal, flow_cfg_step_metal, grouped_gemm_metal,
    joint_attention_metal, rope_3d_metal, short_conv1d_step_metal, temporal_window_attention_metal,
    ternary_gemm_metal, ArtifactAllowlist, MetallibLoader,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::time::Instant;

const WARMUPS: usize = 2;
const SAMPLES: usize = 9;

#[derive(Debug, Serialize)]
struct KernelMeasurement {
    median_us: f64,
    p95_us: f64,
}

#[derive(Debug, Serialize)]
struct BenchmarkRecord {
    schema: &'static str,
    workload: &'static str,
    warmups: usize,
    samples: usize,
    measurements: std::collections::BTreeMap<String, KernelMeasurement>,
    artifacts: std::collections::BTreeMap<String, String>,
}

fn artifact(var: &str) -> (metal_runtime::MetallibArtifact, String) {
    let path = std::env::var(var).unwrap_or_else(|_| panic!("{var} test artifact"));
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    let digest_hex = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    let root = std::path::Path::new(&path).parent().unwrap();
    let name = std::path::Path::new(&path)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();
    let artifact = MetallibLoader::new(root, ArtifactAllowlist::new([(name.to_owned(), digest)]))
        .load(name)
        .unwrap();
    (artifact, digest_hex)
}

fn percentiles(mut samples: Vec<f64>) -> (f64, f64) {
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    let p95 = samples[((samples.len() - 1) * 95) / 100];
    (median, p95)
}

fn report(name: &str, samples: Vec<f64>) -> KernelMeasurement {
    let (median, p95) = percentiles(samples);
    println!("kernel={name} median_us={median:.3} p95_us={p95:.3}");
    KernelMeasurement {
        median_us: median,
        p95_us: p95,
    }
}

#[test]
#[ignore = "explicit artifact-backed performance baseline"]
fn public_bindings_percentiles() {
    let (adaln, adaln_hash) = artifact("ADALN_METALLIB");
    let (flow, flow_hash) = artifact("FLOW_STEP_METALLIB");
    let (joint, joint_hash) = artifact("JOINT_ATTENTION_METALLIB");
    let (rope, rope_hash) = artifact("ROPE_3D_METALLIB");
    let (ternary, ternary_hash) = artifact("TERNARY_GEMM_METALLIB");
    let (temporal, temporal_hash) = artifact("TEMPORAL_ATTN_METALLIB");
    let (diffusion, diffusion_hash) = artifact("DIFFUSION_CONFIDENCE_METALLIB");
    let (short_conv, short_conv_hash) = artifact("SHORT_CONV_METALLIB");
    let (grouped_gemm, grouped_gemm_hash) = artifact("MOE_GROUPED_GEMM_METALLIB");
    let x = [1.0_f32, -2.0, 0.5, 3.0, -1.5, 2.5];
    let scale = [0.1_f32, -0.2, 0.0, 0.3, 0.2, -0.1];
    let shift = [0.5_f32, 0.0, -0.25, 0.1, -0.2, 0.4];
    let uncond = [0.5_f32, 1.0, -1.0, 0.25, 0.0, 0.5];
    let cond = [1.5_f32, 0.0, 2.0, -0.75, 1.0, -0.5];
    let q_joint = [0.5_f32, -1.0, 1.5, 0.25];
    let k_joint = [0.25_f32, 0.5, -0.75, 1.0, 1.25, -0.5];
    let v_joint = [1.0_f32, 2.0, 3.0, 4.0, -1.0, 0.5];
    let positions = [[0_u32, 1, 2, 0], [1, 2, 3, 0]];
    let inv = [0.125_f32];
    let q_rope: Vec<f32> = (0..12).map(|i| i as f32 * 0.25 - 1.0).collect();
    let k_rope: Vec<f32> = q_rope.iter().map(|v| v * 0.5).collect();
    let activations = [1.0_f32, -2.0, 0.5, 3.0, -1.0, -0.5, 2.0, 1.5, 0.0, 4.0];
    let packed_weights = [0x49_u8, 0x02, 0x55, 0x01, 0x14, 0x15];
    let ternary_scales = [0.5_f32, 1.25, 2.0];
    let q_temporal = [1.0_f32, 0.0, 0.5, 1.0, -1.0, 0.25, 0.0, 0.5];
    let k_temporal = [0.5_f32, 1.0, 0.0, 1.0, 1.0, 0.25, -0.5, 0.5];
    let v_temporal = [1.0_f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -2.0];
    let diffusion_logits: Vec<f32> = (0..128).map(|i| (i as f32 * 0.03125).sin()).collect();
    let short_kernel = [1.0_f32, 0.5, -0.25, 0.125];
    let mut short_state = [0.0_f32; 3];
    let moe_activations = [1.0_f32, -0.5, 0.25, 0.75];
    let moe_weights = [0.5_f32, 0.25, -0.75, 1.0, -0.25, 0.5, 0.75, -0.5];
    let moe_tokens = [0_u32, 1];
    let moe_experts = [0_u32, 1];

    for _ in 0..WARMUPS {
        adaln_rms_metal(&x, &scale, &shift, 2, 3, 1e-5, &adaln).unwrap();
        flow_cfg_step_metal(&x, &uncond, &cond, 1.5, 0.1, &flow).unwrap();
        joint_attention_metal(&q_joint, &k_joint, &v_joint, 2, 3, 1, 2, 0.70710677, &joint)
            .unwrap();
        rope_3d_metal(&q_rope, &k_rope, &positions, &inv, &inv, &inv, 1, 6, &rope).unwrap();
        ternary_gemm_metal(
            &activations,
            &packed_weights,
            &ternary_scales,
            2,
            5,
            3,
            &ternary,
        )
        .unwrap();
        temporal_window_attention_metal(
            &q_temporal,
            &k_temporal,
            &v_temporal,
            4,
            1,
            2,
            2,
            0.70710677,
            &temporal,
        )
        .unwrap();
        diffusion_argmax_confidence_metal(&diffusion_logits, 2, 64, &diffusion).unwrap();
        short_conv1d_step_metal(0.75, &short_kernel, &mut short_state, &short_conv).unwrap();
        grouped_gemm_metal(
            &moe_activations,
            &moe_weights,
            &moe_tokens,
            &moe_experts,
            2,
            2,
            &grouped_gemm,
        )
        .unwrap();
    }

    let mut adaln_samples = Vec::with_capacity(SAMPLES);
    let mut flow_samples = Vec::with_capacity(SAMPLES);
    let mut joint_samples = Vec::with_capacity(SAMPLES);
    let mut rope_samples = Vec::with_capacity(SAMPLES);
    let mut ternary_samples = Vec::with_capacity(SAMPLES);
    let mut temporal_samples = Vec::with_capacity(SAMPLES);
    let mut diffusion_samples = Vec::with_capacity(SAMPLES);
    let mut short_conv_samples = Vec::with_capacity(SAMPLES);
    let mut grouped_gemm_samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let started = Instant::now();
        adaln_rms_metal(&x, &scale, &shift, 2, 3, 1e-5, &adaln).unwrap();
        adaln_samples.push(started.elapsed().as_secs_f64() * 1e6);

        let started = Instant::now();
        flow_cfg_step_metal(&x, &uncond, &cond, 1.5, 0.1, &flow).unwrap();
        flow_samples.push(started.elapsed().as_secs_f64() * 1e6);

        let started = Instant::now();
        joint_attention_metal(&q_joint, &k_joint, &v_joint, 2, 3, 1, 2, 0.70710677, &joint)
            .unwrap();
        joint_samples.push(started.elapsed().as_secs_f64() * 1e6);

        let started = Instant::now();
        rope_3d_metal(&q_rope, &k_rope, &positions, &inv, &inv, &inv, 1, 6, &rope).unwrap();
        rope_samples.push(started.elapsed().as_secs_f64() * 1e6);

        let started = Instant::now();
        ternary_gemm_metal(
            &activations,
            &packed_weights,
            &ternary_scales,
            2,
            5,
            3,
            &ternary,
        )
        .unwrap();
        ternary_samples.push(started.elapsed().as_secs_f64() * 1e6);

        let started = Instant::now();
        temporal_window_attention_metal(
            &q_temporal,
            &k_temporal,
            &v_temporal,
            4,
            1,
            2,
            2,
            0.70710677,
            &temporal,
        )
        .unwrap();
        temporal_samples.push(started.elapsed().as_secs_f64() * 1e6);

        let started = Instant::now();
        diffusion_argmax_confidence_metal(&diffusion_logits, 2, 64, &diffusion).unwrap();
        diffusion_samples.push(started.elapsed().as_secs_f64() * 1e6);

        let started = Instant::now();
        short_conv1d_step_metal(0.75, &short_kernel, &mut short_state, &short_conv).unwrap();
        short_conv_samples.push(started.elapsed().as_secs_f64() * 1e6);

        let started = Instant::now();
        grouped_gemm_metal(
            &moe_activations,
            &moe_weights,
            &moe_tokens,
            &moe_experts,
            2,
            2,
            &grouped_gemm,
        )
        .unwrap();
        grouped_gemm_samples.push(started.elapsed().as_secs_f64() * 1e6);
    }

    let measurements = [
        ("adaln_rms_f32", report("adaln_rms_f32", adaln_samples)),
        (
            "flow_cfg_step_f32",
            report("flow_cfg_step_f32", flow_samples),
        ),
        (
            "joint_attention_f32",
            report("joint_attention_f32", joint_samples),
        ),
        ("rope_3d_f32", report("rope_3d_f32", rope_samples)),
        (
            "ternary_gemm_f32",
            report("ternary_gemm_f32", ternary_samples),
        ),
        (
            "temporal_window_attention_f32",
            report("temporal_window_attention_f32", temporal_samples),
        ),
        (
            "diffusion_argmax_confidence_f32",
            report("diffusion_argmax_confidence_f32", diffusion_samples),
        ),
        (
            "short_conv1d_step_f32",
            report("short_conv1d_step_f32", short_conv_samples),
        ),
        (
            "moe_grouped_gemm_f32",
            report("moe_grouped_gemm_f32", grouped_gemm_samples),
        ),
    ]
    .into_iter()
    .map(|(name, measurement)| (name.to_owned(), measurement))
    .collect();
    let artifacts = [
        ("ADALN_METALLIB", adaln_hash),
        ("FLOW_STEP_METALLIB", flow_hash),
        ("JOINT_ATTENTION_METALLIB", joint_hash),
        ("ROPE_3D_METALLIB", rope_hash),
        ("TERNARY_GEMM_METALLIB", ternary_hash),
        ("TEMPORAL_ATTN_METALLIB", temporal_hash),
        ("DIFFUSION_CONFIDENCE_METALLIB", diffusion_hash),
        ("SHORT_CONV_METALLIB", short_conv_hash),
        ("MOE_GROUPED_GEMM_METALLIB", grouped_gemm_hash),
    ]
    .into_iter()
    .map(|(name, hash)| (name.to_owned(), hash))
    .collect();
    let record = BenchmarkRecord {
        schema: "metal-runtime.percentiles.v1",
        workload: "binding-smoke",
        warmups: WARMUPS,
        samples: SAMPLES,
        measurements,
        artifacts,
    };
    if let Ok(path) = std::env::var("METAL_BENCH_OUTPUT") {
        let json = serde_json::to_vec_pretty(&record).unwrap();
        std::fs::write(&path, json).unwrap_or_else(|error| panic!("write {path}: {error}"));
        println!("baseline={path}");
    }
}
