//! (a) Mamba selective scan — `OperatorKind::Scan`, head_dim=8,
//! chunk_size=16. Pins the deterministic selector against three backends
//! (Reference scalar, CPU SIMD with NEON, Metal GPU), the experimental
//! policy against the same registry, and the executor trace shape.

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::selector::{RejectionReason, SelectionDecision};
use kernel_registry::{
    BackendKind, CandidateId, Capability, ExecutionTrace, KernelKey, KernelRegistry,
    SelectionPolicy,
};

use super::{
    build_record, fresh_capabilities, make_candidate, samples_with_p95, shape, NOW_UNIX_MS,
    TEST_FINGERPRINT,
};

pub fn mamba_key() -> KernelKey {
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

pub fn mamba_registry() -> (KernelRegistry, CandidateId, CandidateId, CandidateId) {
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
        build_record(
            id_scalar,
            key.clone(),
            &samples_with_p95(5000),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_simd,
            key.clone(),
            &samples_with_p95(2000),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_metal,
            key.clone(),
            &samples_with_p95(1500),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    (reg, id_scalar, id_simd, id_metal)
}

#[test]
fn mamba_scan_deterministic_picks_lowest_p95_metal_backend() {
    let (reg, _id_scalar, _id_simd, id_metal) = mamba_registry();
    let decision = reg.select_with_caps(
        &mamba_key(),
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
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
            assert!(
                candidate.tunable,
                "ExperimentalOnly must select a tunable candidate; got non-tunable {:?}",
                candidate.name
            );
        }
        other => panic!("expected Chosen under ExperimentalOnly, got {other:?}"),
    }
}

#[test]
fn mamba_scan_trace_lists_chosen_candidate() {
    let (reg, _id_scalar, _id_simd, id_metal) = mamba_registry();
    let decision = reg.select_with_caps(
        &mamba_key(),
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace: ExecutionTrace = reg.explain(&decision);
    assert_eq!(
        trace.selected,
        Some(id_metal),
        "trace must record the chosen candidate id"
    );
    assert!(
        trace.tuning_record_id.is_some(),
        "tuned selection must carry a tuning_record_id"
    );
    assert!(
        trace.human_explanation.contains(&format!("{}", id_metal)),
        "human explanation must mention the chosen id; got {:?}",
        trace.human_explanation
    );
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
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    // The trace lists every considered candidate id (the rejected one).
    let trace = reg.explain(&decision);
    match &decision {
        SelectionDecision::Rejected {
            rejections,
            considered,
        } => {
            assert!(
                considered.contains(&id),
                "candidate must appear in the considered list"
            );
            assert!(
                rejections
                    .iter()
                    .any(|r| matches!(r.reason, RejectionReason::UnsupportedDtype(_))),
                "expected UnsupportedDtype rejection, got {rejections:?}"
            );
        }
        other => panic!("expected Rejected, got {other:?}"),
    }
    let trace_ids: Vec<CandidateId> = trace.considered.iter().map(|r| r.candidate).collect();
    assert!(
        trace_ids.contains(&id),
        "ExecutionTrace.considered must list every rejected candidate id; got {trace_ids:?}"
    );
    assert!(
        trace.human_explanation.to_lowercase().contains("dtype"),
        "human explanation should categorize the rejection; got {:?}",
        trace.human_explanation
    );
}
