//! (h) LLaDA / Dream diffusion — OperatorKind::Diffusion, vocab=16, steps=8

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::selector::SelectionDecision;
use kernel_registry::{BackendKind, Capability, KernelKey, KernelRegistry, SelectionPolicy};

use super::{
    build_record, fresh_capabilities, make_candidate, samples_with_p95, shape, NOW_UNIX_MS,
    TEST_FINGERPRINT,
};

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
        build_record(
            id_scalar,
            key.clone(),
            &samples_with_p95(9000),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_metal,
            key.clone(),
            &samples_with_p95(2200),
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
                "metal p95=2200 must beat scalar p95=9000"
            );
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
        build_record(
            id_metal,
            key.clone(),
            &samples_with_p95(2200),
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
