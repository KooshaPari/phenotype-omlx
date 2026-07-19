//! Candidate capability and shape-range filtering contracts.

use kernel_registry::compat::OperatorKind;
use kernel_registry::selector::{RejectionReason, SelectionDecision};
use kernel_registry::{
    BackendKind, Candidate, CandidateId, Capability, DeviceCaps, KernelRegistry, SelectionPolicy,
};

use super::{
    candidate_from, key_with, shape_with, DType, NOW_UNIX_MS, TEST_DEVICE_FINGERPRINT,
};

#[test]
fn candidate_capability_filter_rejects_when_required_capability_missing() {
    let mut reg = KernelRegistry::new();
    let cand = candidate_from(
        "metal-gemm-fp16",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
    );
    let id = cand.id;
    reg.register_candidate(cand);

    let key = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);

    // Missing MetalGpu — every candidate must be rejected.
    let caps = DeviceCaps { capabilities: vec![Capability::Bf16] };
    // The selector isn't directly exposed; we instead assert that
    // `register_candidate` succeeded and that `select` with a no-metal
    // device rejects the candidate with a MissingCapability reason.
    let mut registry = KernelRegistry::new();
    registry.register_candidate(Candidate {
        id,
        ..candidate_from("metal-gemm-fp16", BackendKind::Metal, vec![Capability::MetalGpu])
    });
    let decision = registry.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &caps,
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Rejected { rejections, considered } => {
            assert!(considered.contains(&id),
                "the candidate must appear in the considered list");
            assert!(rejections.iter().any(|r| matches!(r.reason, RejectionReason::MissingCapability(_))),
                "expected MissingCapability reason, got {rejections:?}");
        }
        other => panic!("expected Rejected, got {other:?}"),
    }
    // Silence unused-mut warning when the local `reg` is dropped.
    let _ = &mut reg;
}

#[test]
fn candidate_shape_range_filter_rejects_out_of_range() {
    let mut reg = KernelRegistry::new();
    let cand = Candidate {
        id: CandidateId(1),
        name: "shape-bounded".into(),
        backend: BackendKind::Metal,
        source_hash: "sha256:shape-bounded".into(),
        requires: vec![Capability::MetalGpu],
        min_shape: shape_with(2, 2, 2, 1, 1, 1),
        max_shape: shape_with(128, 128, 128, 4, 128, 4),
        supports_dtypes: vec![DType::Fp16],
        tunable: true,
    };
    let id = cand.id;
    reg.register_candidate(cand);

    let mut huge_key = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    huge_key.shape_signature = shape_with(1024, 1024, 1024, 1, 1024, 1);

    let caps = DeviceCaps { capabilities: vec![Capability::MetalGpu] };
    let decision = reg.select_with_caps(
        &huge_key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &caps,
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Rejected { rejections, considered } => {
            assert!(considered.contains(&id));
            assert!(rejections.iter().any(|r| matches!(r.reason, RejectionReason::ShapeOutOfRange)),
                "expected ShapeOutOfRange, got {rejections:?}");
        }
        other => panic!("expected Rejected, got {other:?}"),
    }
}