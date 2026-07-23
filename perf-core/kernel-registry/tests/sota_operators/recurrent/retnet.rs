//! RetNet retention-step selector coverage.

use kernel_registry::compat::DType;
use kernel_registry::selector::SelectionDecision;
use kernel_registry::{retnet_key, BackendKind, Capability, KernelRegistry, SelectionPolicy};

use super::{
    build_record, fresh_capabilities, make_candidate, samples_with_p95, shape, NOW_UNIX_MS,
    TEST_FINGERPRINT,
};

#[test]
fn retnet_builder_selects_metal_retention_candidate() {
    let key = retnet_key(2, 4, 16, 8, DType::Bf16, TEST_FINGERPRINT, 1);
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 8, 256, 16);
    let reference = make_candidate(
        "RetNetReference",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "RetNetMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let metal_id = metal.id;
    let mut registry = KernelRegistry::new();
    registry.register_candidate(reference);
    registry.register_candidate(metal);
    registry.attach_tuning_record(
        key.clone(),
        build_record(
            metal_id,
            key.clone(),
            &samples_with_p95(900),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );

    let decision = registry.select_with_caps(
        &key,
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => assert_eq!(candidate.id, metal_id),
        other => panic!("expected RetNet Metal candidate, got {other:?}"),
    }
}
