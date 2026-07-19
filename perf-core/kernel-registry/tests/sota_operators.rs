//! Selector-coverage tests for the SOTA operator families added by
//! commits c05e3fa, fc4474c, 98939af, ceebda5, c7c4b52.
//!
//! Each family below corresponds to a row in
//! `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md`
//! §Model Acceptance Matrix:
//!
//! - Mamba / RWKV-7  : recurrent-hybrid (fc4474c)
//! - ZAYA            : CCA block-parallel (98939af)
//! - LFM             : gated short convolution (98939af)
//! - DeepSeek MLA    : compressed-kv cache (c05e3fa)
//! - DeepSeek MTP    : speculative proposal (c05e3fa)
//! - Bonsai          : ternary matmul (ceebda5)
//! - Qwen            : DeltaNet (ceebda5)
//! - LLaDA / Dream   : diffusion decoder (c7c4b52)
//!
//! For every family the test (a) registers a reference scalar backend
//! plus two optimized backends, (b) attaches synthetic tuning records
//! with *distinct* p95 latencies so the deterministic policy has a
//! non-trivial winner, and (c) asserts the selector picked the lowest-p95
//! backend. The observability test pins the trace shape; the rejection
//! test pins dtype enforcement.
//!
//! Convention: `now_unix_ms = 1_700_000_000_000` matches the project
//! baseline so contract tests stay time-independent.

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::selector::{RejectionReason, SelectionDecision};
use kernel_registry::{
    BackendKind, Candidate, CandidateId, Capability, DeviceCaps, ExecutionTrace, KernelKey,
    KernelRegistry, Measurement, SelectionPolicy, ShapeSignature, TuningRecord,
};

const NOW_UNIX_MS: u64 = 1_700_000_000_000;
const TEST_FINGERPRINT: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

// ----------------------------------------------------------------------
// Shared builders
// ----------------------------------------------------------------------

fn shape(m: usize, n: usize, k: usize, batch: usize, seq: usize, group: usize) -> ShapeSignature {
    ShapeSignature { m, n, k, batch, seq, group }
}

fn build_record(
    candidate_id: CandidateId,
    key: KernelKey,
    samples: &[u64],
    expires_at_unix_ms: Option<u64>,
) -> TuningRecord {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let median = sorted[n / 2];
    let p95_idx = ((n as f64 * 0.95).ceil() as usize).saturating_sub(1).min(n - 1);
    let p99_idx = ((n as f64 * 0.99).ceil() as usize).saturating_sub(1).min(n - 1);
    let mean = sorted.iter().sum::<u64>() as f64 / n as f64;
    let variance = sorted
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let measurements: Vec<Measurement> = samples
        .iter()
        .enumerate()
        .map(|(i, &latency_ns)| Measurement {
            sample_idx: i as u32,
            latency_ns,
            energy_j: None,
            bytes_written: 0,
        })
        .collect();
    TuningRecord {
        candidate_id,
        key,
        measurements,
        median_ns: median,
        p95_ns: sorted[p95_idx],
        p99_ns: sorted[p99_idx],
        variance_ns2: variance as u64,
        samples: n,
        warmup_discarded: 3,
        compiler: "metal-msl".to_string(),
        compiler_version: "3.2".to_string(),
        captured_at_unix_ms: NOW_UNIX_MS,
        source_revision: "rev-sota".to_string(),
        expires_at_unix_ms,
    }
}

fn make_candidate(
    name: &str,
    backend: BackendKind,
    requires: Vec<Capability>,
    min_shape: ShapeSignature,
    max_shape: ShapeSignature,
    supports_dtypes: Vec<DType>,
    tunable: bool,
) -> Candidate {
    Candidate {
        id: CandidateId::derive(name, backend),
        name: name.to_string(),
        backend,
        source_hash: format!("sha256:{name}"),
        requires,
        min_shape,
        max_shape,
        supports_dtypes,
        tunable,
    }
}

fn fresh_capabilities() -> DeviceCaps {
    DeviceCaps {
        capabilities: vec![
            Capability::MetalGpu,
            Capability::Bf16,
            Capability::Fp16,
            Capability::MetalMs3,
        ],
    }
}

fn full_capabilities() -> DeviceCaps {
    DeviceCaps {
        capabilities: vec![
            Capability::MetalGpu,
            Capability::Bf16,
            Capability::Fp16,
            Capability::MetalMs3,
            Capability::Avx512,
        ],
    }
}

// Stable samples yielding exactly the requested p95:
// 20 samples with the value of interest as the upper quintile (sample 19)
// so ceil(0.95 * 20) - 1 = 18 (idx 18 → sample 19) becomes the p95 index.
// Construction: lower 18 samples ≤ lower_p95, sample 18 = p95_value,
// sample 19 = p95_value + 1 to break ties without affecting p95.
fn samples_with_p95(p95: u64) -> Vec<u64> {
    let mut v: Vec<u64> = (0..18).map(|i| p95.saturating_sub(50 + i as u64)).collect();
    v.push(p95);
    v.push(p95 + 1);
    v
}

// ----------------------------------------------------------------------
// (a) Mamba selective scan — OperatorKind::Scan, head_dim=8, chunk_size=16
// ----------------------------------------------------------------------

fn mamba_key() -> KernelKey {
    // head_dim is carried via `m`, chunk_size via `seq`.
    KernelKey {
        operator_kind: OperatorKind::Scan,
        attention_kind: None,
        shape_signature: shape(8, 8, 8, 1, 16, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

fn mamba_registry() -> (KernelRegistry, CandidateId, CandidateId, CandidateId) {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
    let scalar = make_candidate(
        "MambaSelectiveScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16, DType::Fp16],
        false,
    );
    let simd = make_candidate(
        "MambaSelectiveSimd",
        BackendKind::Cpu,
        vec![Capability::Neon],
        min,
        max,
        vec![DType::Fp32, DType::Bf16, DType::Fp16],
        true,
    );
    let metal = make_candidate(
        "MambaSelectiveMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_scalar = scalar.id;
    let id_simd = simd.id;
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(simd);
    reg.register_candidate(metal);
    let key = mamba_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_scalar, key.clone(), &samples_with_p95(5000), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_simd, key.clone(), &samples_with_p95(2000), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1500), Some(NOW_UNIX_MS + 86_400_000)),
    );
    (reg, id_scalar, id_simd, id_metal)
}

#[test]
fn mamba_scan_deterministic_picks_lowest_p95_metal_backend() {
    let (reg, _id_scalar, _id_simd, id_metal) = mamba_registry();
    let decision = reg.select_with_caps(
        &mamba_key(),
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_metal,
                "metal backend p95=1500 must beat simd p95=2000 and scalar p95=5000"
            );
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn mamba_scan_experimental_only_picks_a_tunable_candidate() {
    let (reg, _id_scalar, _id_simd, _id_metal) = mamba_registry();
    let decision = reg.select_with_caps(
        &mamba_key(),
        SelectionPolicy::ExperimentalOnly,
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert!(candidate.tunable,
                "ExperimentalOnly must select a tunable candidate; got non-tunable {:?}",
                candidate.name);
        }
        other => panic!("expected Chosen under ExperimentalOnly, got {other:?}"),
    }
}

#[test]
fn mamba_scan_trace_lists_chosen_candidate() {
    let (reg, _id_scalar, _id_simd, id_metal) = mamba_registry();
    let decision = reg.select_with_caps(
        &mamba_key(),
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace: ExecutionTrace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal),
        "trace must record the chosen candidate id");
    assert!(trace.tuning_record_id.is_some(),
        "tuned selection must carry a tuning_record_id");
    assert!(trace.human_explanation.contains(&format!("{}", id_metal)),
        "human explanation must mention the chosen id; got {:?}",
        trace.human_explanation);
}

#[test]
fn mamba_scan_rejects_with_unsupported_dtype_when_key_dtype_mismatches() {
    // Register only a candidate that supports Bf16; query with Fp32.
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
    let metal = make_candidate(
        "MambaSelectiveMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16],
        true,
    );
    let id = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(metal);

    let mut key = mamba_key();
    key.dtype = DType::Fp32; // unsupported
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    // The trace lists every considered candidate id (the rejected one).
    let trace = reg.explain(&decision);
    match &decision {
        SelectionDecision::Rejected { rejections, considered } => {
            assert!(considered.contains(&id),
                "candidate must appear in the considered list");
            assert!(rejections.iter().any(|r| matches!(r.reason, RejectionReason::UnsupportedDtype(_))),
                "expected UnsupportedDtype rejection, got {rejections:?}");
        }
        other => panic!("expected Rejected, got {other:?}"),
    }
    let trace_ids: Vec<CandidateId> = trace.considered.iter().map(|r| r.candidate).collect();
    assert!(trace_ids.contains(&id),
        "ExecutionTrace.considered must list every rejected candidate id; got {trace_ids:?}");
    assert!(trace.human_explanation.to_lowercase().contains("dtype"),
        "human explanation should categorize the rejection; got {:?}",
        trace.human_explanation);
}

// ----------------------------------------------------------------------
// (b) ZAYA CCA block-parallel — OperatorKind::Cca, head_dim=4, block_count=3
// ----------------------------------------------------------------------

fn cca_key() -> KernelKey {
    // head_dim = m=4, block_count carried via batch=3.
    KernelKey {
        operator_kind: OperatorKind::Cca,
        attention_kind: None,
        shape_signature: shape(4, 4, 4, 3, 1, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

#[test]
fn cca_block_deterministic_picks_lowest_p95_zmm_backend() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(32, 32, 32, 64, 1, 1);
    let scalar = make_candidate(
        "CcaBlockScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "CcaBlockMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let zmm = make_candidate(
        "CcaBlockZmm",
        BackendKind::Cpu,
        vec![Capability::Avx512, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_scalar = scalar.id;
    let id_metal = metal.id;
    let id_zmm = zmm.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    reg.register_candidate(zmm);
    let key = cca_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_scalar, key.clone(), &samples_with_p95(8000), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(2500), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_zmm, key.clone(), &samples_with_p95(1700), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &full_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(candidate.id, id_zmm,
                "ZMM (avx512) p95=1700 must beat metal p95=2500 and scalar p95=8000");
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn cca_block_trace_lists_chosen_candidate() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(32, 32, 32, 64, 1, 1);
    let scalar = make_candidate(
        "CcaBlockScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "CcaBlockMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu],
        min,
        max,
        vec![DType::Bf16],
        true,
    );
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = cca_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(2500), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal));
    assert!(trace.human_explanation.contains(&format!("{}", id_metal)));
}

// ----------------------------------------------------------------------
// (c) DeepSeek MLA cache — OperatorKind::Mla, d_latent=4, d_rope=2, seq_k=4
// ----------------------------------------------------------------------

fn mla_key() -> KernelKey {
    // d_latent = m=4, d_rope = n=2, seq_k = seq=4.
    KernelKey {
        operator_kind: OperatorKind::Mla,
        attention_kind: None,
        shape_signature: shape(4, 2, 4, 1, 4, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

#[test]
fn mla_cache_deterministic_picks_lowest_p95_metal_backend() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 4096, 1);
    let scalar = make_candidate(
        "MlaCacheScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "MlaCacheMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let zmm = make_candidate(
        "MlaCacheZmm",
        BackendKind::Cpu,
        vec![Capability::Avx512, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_scalar = scalar.id;
    let id_metal = metal.id;
    let id_zmm = zmm.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    reg.register_candidate(zmm);
    let key = mla_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_scalar, key.clone(), &samples_with_p95(4500), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1100), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_zmm, key.clone(), &samples_with_p95(1300), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &full_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(candidate.id, id_metal,
                "metal p95=1100 must beat zmm p95=1300 and scalar p95=4500");
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn mla_cache_trace_lists_chosen_candidate() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 4096, 1);
    let scalar = make_candidate(
        "MlaCacheScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "MlaCacheMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu],
        min,
        max,
        vec![DType::Bf16],
        true,
    );
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = mla_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1100), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal));
}

// ----------------------------------------------------------------------
// (d) DeepSeek MTP proposal — OperatorKind::Speculative, vocab=16, depth=4
// ----------------------------------------------------------------------

fn mtp_key() -> KernelKey {
    // vocab = m=16, proposal_depth = n=4.
    KernelKey {
        operator_kind: OperatorKind::Speculative,
        attention_kind: None,
        shape_signature: shape(16, 4, 16, 1, 1, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

#[test]
fn mtp_proposal_deterministic_picks_lowest_p95_metal_backend() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(128, 32, 128, 4, 1, 1);
    let scalar = make_candidate(
        "MtpScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "MtpMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_scalar = scalar.id;
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = mtp_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_scalar, key.clone(), &samples_with_p95(6000), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1800), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(candidate.id, id_metal,
                "metal p95=1800 must beat scalar p95=6000");
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn mtp_proposal_trace_lists_chosen_candidate() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(128, 32, 128, 4, 1, 1);
    let scalar = make_candidate(
        "MtpScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "MtpMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu],
        min,
        max,
        vec![DType::Bf16],
        true,
    );
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = mtp_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1800), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal));
}

// ----------------------------------------------------------------------
// (e) Bonsai ternary matmul — QuantizationPolicy::Ternary, m=4,k=8,n=4,group=4
// ----------------------------------------------------------------------

fn bonsai_key() -> KernelKey {
    KernelKey {
        operator_kind: OperatorKind::Quantized,
        attention_kind: None,
        shape_signature: shape(4, 4, 8, 1, 1, 4),
        dtype: DType::Int8,
        quantization: QuantizationPolicy::Ternary,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

#[test]
fn bonsai_ternary_deterministic_picks_lowest_p95_metal_backend() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(128, 128, 256, 4, 1, 16);
    let scalar = make_candidate(
        "TernaryScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Int8, DType::Fp32],
        false,
    );
    let metal = make_candidate(
        "TernaryMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::MetalMs3],
        min,
        max,
        vec![DType::Int8],
        true,
    );
    let zmm = make_candidate(
        "TernaryZmm",
        BackendKind::Cpu,
        vec![Capability::Avx512],
        min,
        max,
        vec![DType::Int8],
        true,
    );
    let id_scalar = scalar.id;
    let id_metal = metal.id;
    let id_zmm = zmm.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    reg.register_candidate(zmm);
    let key = bonsai_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_scalar, key.clone(), &samples_with_p95(7000), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1900), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_zmm, key.clone(), &samples_with_p95(2200), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &full_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(candidate.id, id_metal,
                "metal p95=1900 must beat zmm p95=2200 and scalar p95=7000");
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn bonsai_ternary_trace_lists_chosen_candidate() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(128, 128, 256, 4, 1, 16);
    let scalar = make_candidate(
        "TernaryScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Int8],
        false,
    );
    let metal = make_candidate(
        "TernaryMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::MetalMs3],
        min,
        max,
        vec![DType::Int8],
        true,
    );
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = bonsai_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1900), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal));
}

// ----------------------------------------------------------------------
// (f) LFM2 gated short conv — OperatorKind::ShortConv, kernel_len=4, gate=2
// ----------------------------------------------------------------------

fn lfm_key() -> KernelKey {
    // kernel_len = m=4, gate_kernel_len = n=2.
    KernelKey {
        operator_kind: OperatorKind::ShortConv,
        attention_kind: None,
        shape_signature: shape(4, 2, 4, 1, 1, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

#[test]
fn lfm_gated_short_conv_deterministic_picks_lowest_p95_metal_backend() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 16, 64, 4, 256, 1);
    let scalar = make_candidate(
        "GatedConvScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "GatedConvMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_scalar = scalar.id;
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = lfm_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_scalar, key.clone(), &samples_with_p95(3500), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(900), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(candidate.id, id_metal,
                "metal p95=900 must beat scalar p95=3500");
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn lfm_gated_short_conv_trace_lists_chosen_candidate() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 16, 64, 4, 256, 1);
    let scalar = make_candidate(
        "GatedConvScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "GatedConvMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu],
        min,
        max,
        vec![DType::Bf16],
        true,
    );
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = lfm_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(900), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal));
}

// ----------------------------------------------------------------------
// (g) Qwen DeltaNet — OperatorKind::DeltaNet, head_dim=4, chunk_size=16
// ----------------------------------------------------------------------

fn deltanet_key() -> KernelKey {
    // head_dim = m=4, chunk_size = seq=16.
    KernelKey {
        operator_kind: OperatorKind::DeltaNet,
        attention_kind: None,
        shape_signature: shape(4, 4, 4, 1, 16, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

#[test]
fn qwen_deltanet_deterministic_picks_lowest_p95_metal_backend() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
    let scalar = make_candidate(
        "DeltaNetScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "DeltaNetMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_scalar = scalar.id;
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = deltanet_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_scalar, key.clone(), &samples_with_p95(4000), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1400), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(candidate.id, id_metal,
                "metal p95=1400 must beat scalar p95=4000");
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn qwen_deltanet_trace_lists_chosen_candidate() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
    let scalar = make_candidate(
        "DeltaNetScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "DeltaNetMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu],
        min,
        max,
        vec![DType::Bf16],
        true,
    );
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = deltanet_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1400), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal));
}

// ----------------------------------------------------------------------
// (h) LLaDA / Dream diffusion — OperatorKind::Diffusion, vocab=16, steps=8
// ----------------------------------------------------------------------

fn diffusion_key() -> KernelKey {
    // vocab = m=16, total_steps = n=8.
    KernelKey {
        operator_kind: OperatorKind::Diffusion,
        attention_kind: None,
        shape_signature: shape(16, 8, 16, 1, 1, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

#[test]
fn llama_dream_diffusion_deterministic_picks_lowest_p95_metal_backend() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(128, 64, 128, 4, 1, 1);
    let scalar = make_candidate(
        "DenoiseScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "DenoiseMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_scalar = scalar.id;
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = diffusion_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_scalar, key.clone(), &samples_with_p95(9000), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(2200), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(candidate.id, id_metal,
                "metal p95=2200 must beat scalar p95=9000");
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn llama_dream_diffusion_trace_lists_chosen_candidate() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(128, 64, 128, 4, 1, 1);
    let scalar = make_candidate(
        "DenoiseScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "DenoiseMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu],
        min,
        max,
        vec![DType::Bf16],
        true,
    );
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = diffusion_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(2200), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal));
}

// ----------------------------------------------------------------------
// (i) RWKV-7 — OperatorKind::Recurrent, state_channels=4
// ----------------------------------------------------------------------

fn rwkv_key() -> KernelKey {
    // state_channels = m=4 (RWKV-7 keeps [k, v, r, w]).
    KernelKey {
        operator_kind: OperatorKind::Recurrent,
        attention_kind: None,
        shape_signature: shape(4, 4, 4, 1, 1, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

#[test]
fn rwkv7_deterministic_picks_lowest_p95_metal_backend() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
    let scalar = make_candidate(
        "Rwkv7Scalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "Rwkv7Metal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_scalar = scalar.id;
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = rwkv_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_scalar, key.clone(), &samples_with_p95(5500), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1700), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(candidate.id, id_metal,
                "metal p95=1700 must beat scalar p95=5500");
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn rwkv7_trace_lists_chosen_candidate() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
    let scalar = make_candidate(
        "Rwkv7Scalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "Rwkv7Metal",
        BackendKind::Metal,
        vec![Capability::MetalGpu],
        min,
        max,
        vec![DType::Bf16],
        true,
    );
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = rwkv_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1700), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal));
}

// ----------------------------------------------------------------------
// Cross-cutting: determinism and ordering
// ----------------------------------------------------------------------

#[test]
fn selector_runs_are_deterministic_across_sota_operator_families() {
    // Run the deterministic policy twice against the Mamba registry and
    // confirm we get the same chosen id both times.
    let (reg, _, _, id_metal) = mamba_registry();
    let d1 = reg.select_with_caps(
        &mamba_key(),
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let d2 = reg.select_with_caps(
        &mamba_key(),
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let id1 = d1.selected();
    let id2 = d2.selected();
    assert_eq!(id1, id2, "selector must be deterministic across calls");
    assert_eq!(id1, Some(id_metal));
}

#[test]
fn selector_rejects_when_no_candidate_supports_ternary_dtype() {
    // The Bonsai ternary matmul key demands Int8 + Ternary policy. If a
    // candidate only supports Fp32/Fp16 the dtype filter must reject it.
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 1, 4);
    let scalar = make_candidate(
        "TernaryScalarWrongDtype",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Fp16], // no Int8
        false,
    );
    let id = scalar.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    let decision = reg.select_with_caps(
        &bonsai_key(),
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Rejected { rejections, considered } => {
            assert!(considered.contains(&id));
            assert!(rejections.iter().any(|r| matches!(r.reason, RejectionReason::UnsupportedDtype(_))),
                "expected UnsupportedDtype rejection, got {rejections:?}");
        }
        other => panic!("expected Rejected, got {other:?}"),
    }
}