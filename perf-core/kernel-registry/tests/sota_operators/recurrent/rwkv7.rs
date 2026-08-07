//! (i) RWKV-7 — `OperatorKind::Recurrent`, state_channels=4. Plus the
//! batched DeltaNet coverage: the Qwen3-Coder-Next style hybrid DeltaNet
//! dispatches a *parallel* implementation for shape signatures with
//! batch >= 2. The selector must surface at least one
//! `DeltaNetBatched`-tagged candidate for the (B=2, H=2, C=4, D=8)
//! signature so the runtime can pick the parallel kernel.

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::selector::SelectionDecision;
use kernel_registry::{BackendKind, Capability, KernelKey, KernelRegistry, SelectionPolicy};

use super::{
    build_record, fresh_capabilities, make_candidate, samples_with_p95, shape, NOW_UNIX_MS,
    TEST_FINGERPRINT,
};

// (i) RWKV-7 — OperatorKind::Recurrent, state_channels=4

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
        build_record(
            id_scalar,
            key.clone(),
            &samples_with_p95(5500),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_metal,
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
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_metal,
                "metal p95=1700 must beat scalar p95=5500"
            );
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
        build_record(
            id_metal,
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
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal));
}

// (h) Qwen DeltaNet *batched* — covers `(B=2, H=2, C=4, D=8)` shape
// signatures for Qwen3-Coder-Next style hybrid DeltaNet. The selector
// must return at least one candidate tagged `DeltaNetBatched` for this
// shape so the runtime can dispatch the parallel implementation
// instead of the single-(batch, head) chunk.

fn deltanet_batched_key() -> KernelKey {
    // (B=2, H=2, C=4, D=8) carried via (m=D=8, n=D=8, k=D=8, batch=B=2,
    // seq=C=4, group=1 — heads are not GQA groups for DeltaNet). The
    // batched shape signature is what the runtime queries when
    // Qwen3-Coder-Next dispatches the parallel DeltaNet path.
    KernelKey {
        operator_kind: OperatorKind::DeltaNet,
        attention_kind: None,
        shape_signature: shape(8, 8, 8, 2, 4, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

#[test]
fn deltanet_batched_selector_returns_tagged_candidate() {
    // Register a Reference and a Metal candidate for DeltaNetBatched.
    // The Metal candidate name carries the `DeltaNetBatched` tag so the
    // runtime can dispatch by tag. The selector must pick the Metal
    // candidate (lowest p95) and the chosen candidate's name must
    // contain `DeltaNetBatched`.
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
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
    let id_scalar = scalar.id;
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = deltanet_batched_key();
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
            assert_eq!(
                candidate.id, id_metal,
                "metal p95=1300 must beat scalar p95=4500"
            );
            assert!(
                candidate.name.contains("DeltaNetBatched"),
                "chosen candidate must be tagged DeltaNetBatched; got {:?}",
                candidate.name
            );
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn deltanet_batched_considered_list_contains_tagged_candidate() {
    // Confirms at least one candidate tagged `DeltaNetBatched` appears
    // in the selector's considered list (regardless of which the
    // policy ultimately picks). This is the "selector sees the new
    // kernel" assertion the spec asks for.
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
    let scalar = make_candidate(
        "DeltaNetBatchedScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Bf16],
        false,
    );
    let id_scalar = scalar.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    let key = deltanet_batched_key();
    // No tuning records: the Deterministic policy will fall back to the
    // Reference backend. The test only asserts the selector saw the
    // tagged candidate.
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let tagged_seen = match &decision {
        SelectionDecision::Chosen { candidate, .. } => candidate.name.contains("DeltaNetBatched"),
        SelectionDecision::Rejected { considered, .. } => considered.contains(&id_scalar),
    };
    assert!(
        tagged_seen,
        "selector must surface at least one DeltaNetBatched-tagged candidate for the (B=2,H=2,C=4,D=8) signature; got {decision:?}"
    );
}
