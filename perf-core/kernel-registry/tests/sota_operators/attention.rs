//! (b) ZAYA CCA block-parallel — OperatorKind::Cca, head_dim=4, block_count=3
//! (c) DeepSeek MLA cache   — OperatorKind::Mla, d_latent=4, d_rope=2, seq_k=4
//! (d) DeepSeek MTP proposal — OperatorKind::Speculative, vocab=16, depth=4

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::selector::SelectionDecision;
use kernel_registry::{BackendKind, Capability, KernelKey, KernelRegistry, SelectionPolicy};

use super::{
    build_record, fresh_capabilities, full_capabilities, make_candidate, samples_with_p95, shape,
    NOW_UNIX_MS, TEST_FINGERPRINT,
};

// (b) ZAYA CCA block-parallel — OperatorKind::Cca, head_dim=4, block_count=3

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
        build_record(
            id_scalar,
            key.clone(),
            &samples_with_p95(8000),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_metal,
            key.clone(),
            &samples_with_p95(2500),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_zmm,
            key.clone(),
            &samples_with_p95(1700),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &full_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_zmm,
                "ZMM (avx512) p95=1700 must beat metal p95=2500 and scalar p95=8000"
            );
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
        build_record(
            id_metal,
            key.clone(),
            &samples_with_p95(2500),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal));
    assert!(trace.human_explanation.contains(&format!("{}", id_metal)));
}

// (c) DeepSeek MLA cache — OperatorKind::Mla, d_latent=4, d_rope=2, seq_k=4

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
        build_record(
            id_scalar,
            key.clone(),
            &samples_with_p95(4500),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_metal,
            key.clone(),
            &samples_with_p95(1100),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_zmm,
            key.clone(),
            &samples_with_p95(1300),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &full_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_metal,
                "metal p95=1100 must beat zmm p95=1300 and scalar p95=4500"
            );
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
        build_record(
            id_metal,
            key.clone(),
            &samples_with_p95(1100),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal));
}

// (d) DeepSeek MTP proposal — OperatorKind::Speculative, vocab=16, depth=4

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
        build_record(
            id_scalar,
            key.clone(),
            &samples_with_p95(6000),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_metal,
            key.clone(),
            &samples_with_p95(1800),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    let decision = reg.select_with_caps(
        &key,
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
                "metal p95=1800 must beat scalar p95=6000"
            );
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
        build_record(
            id_metal,
            key.clone(),
            &samples_with_p95(1800),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal));
}
