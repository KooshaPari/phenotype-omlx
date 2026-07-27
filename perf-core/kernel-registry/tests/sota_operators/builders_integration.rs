//! End-to-end integration tests for the [`crate::builders`] module.
//!
//! These tests exercise the bridge from operator-plan style inputs
//! (sliding-window GQA, batched DeltaNet, single-chunk DeltaNet) to a
//! [`KernelKey`] that the selector can resolve against a real
//! registry with registered candidates. The point is to prove that the
//! builders don't just construct plausible bytes — they produce keys
//! the selector actually understands.

use kernel_registry::compat::DType;
use kernel_registry::selector::SelectionDecision;
use kernel_registry::{
    deltanet_batched_key, deltanet_key, sliding_window_key, BackendKind, Capability,
    KernelRegistry, SelectionPolicy,
};

use super::{
    build_record, fresh_capabilities, make_candidate, samples_with_p95, shape, NOW_UNIX_MS,
    TEST_FINGERPRINT,
};

#[test]
fn builder_sliding_window_selector_picks_tagged_metal_candidate() {
    let min = shape(32, 1, 1, 1, 1, 1);
    let max = shape(128, 8, 8, 4, 256, 8);
    let scalar = make_candidate(
        "SlidingWindowScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "SlidingWindowMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);

    // Build the key through the public builder rather than hand-rolling
    // the shape signature — that's the entire point of the bridge.
    let key = sliding_window_key(8, 2, 64, 1, 8, 4, 4, DType::Bf16, TEST_FINGERPRINT, 1);

    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_metal,
            key.clone(),
            &samples_with_p95(900),
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
            assert_eq!(candidate.id, id_metal);
            assert!(
                candidate.name.contains("SlidingWindow"),
                "builder key must route to a SlidingWindow candidate; got {:?}",
                candidate.name
            );
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn builder_deltanet_batched_selector_picks_tagged_metal_candidate() {
    // The builder pins group=num_heads (see spec), so widen the max_shape
    // bound on `group` to accommodate (B=2, H=2, C=4, D=8).
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 16);
    let scalar = make_candidate(
        "DeltaNetBatchedScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "DeltaNetBatchedMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);

    let key = deltanet_batched_key(2, 2, 4, 8, DType::Bf16, TEST_FINGERPRINT, 1);

    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_metal,
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
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(candidate.id, id_metal);
            assert!(
                candidate.name.contains("DeltaNetBatched"),
                "builder key must route to a DeltaNetBatched candidate; got {:?}",
                candidate.name
            );
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn builder_deltanet_selector_picks_tagged_metal_candidate() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
    let scalar = make_candidate(
        "MambaSelectiveScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
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
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);

    let key = deltanet_key(8, 16, DType::Bf16, TEST_FINGERPRINT, 1);

    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_metal,
            key.clone(),
            &samples_with_p95(1500),
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
            assert_eq!(candidate.id, id_metal);
            assert!(
                candidate.name.contains("MambaSelective"),
                "builder key must route to a MambaSelective candidate; got {:?}",
                candidate.name
            );
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}
