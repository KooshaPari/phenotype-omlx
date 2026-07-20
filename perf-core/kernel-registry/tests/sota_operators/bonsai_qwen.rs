//! (e) Bonsai ternary matmul — QuantizationPolicy::Ternary, m=4,k=8,n=4,group=4
//! (f) LFM2 gated short conv — OperatorKind::ShortConv, kernel_len=4, gate=2
//! (g) Qwen DeltaNet       — OperatorKind::DeltaNet, head_dim=4, chunk_size=16

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::selector::SelectionDecision;
use kernel_registry::{
    BackendKind, Capability, KernelKey, KernelRegistry, SelectionPolicy,
};

use super::{
    build_record, fresh_capabilities, full_capabilities, make_candidate, samples_with_p95, shape,
    NOW_UNIX_MS, TEST_FINGERPRINT,
};

// (e) Bonsai ternary matmul — QuantizationPolicy::Ternary, m=4,k=8,n=4,group=4

pub(super) fn bonsai_key() -> KernelKey {
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

// (f) LFM2 gated short conv — OperatorKind::ShortConv, kernel_len=4, gate=2

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

// (g) Qwen DeltaNet — OperatorKind::DeltaNet, head_dim=4, chunk_size=16

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