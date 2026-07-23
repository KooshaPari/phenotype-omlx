//! (e) Qwen3-Next sliding-window GQA attention (long-context).
//!
//! Pinned shape: seq_q=8, q_heads=8, kv_heads=2, head_dim=64,
//! group_size=4, window_size=4 — the canonical "3:1 grouped + 4k
//! sliding" Qwen3-Next layer configuration.

use kernel_registry::compat::{AttentionKind, DType, OperatorKind, QuantizationPolicy};
use kernel_registry::selector::SelectionDecision;
use kernel_registry::{BackendKind, Capability, KernelKey, KernelRegistry, SelectionPolicy};

use super::{
    build_record, fresh_capabilities, make_candidate, samples_with_p95, shape, NOW_UNIX_MS,
    TEST_FINGERPRINT,
};

fn sliding_window_key() -> KernelKey {
    // m = q_heads * head_dim / 8 = 8 * 64 / 8 = 64 (packed dimensions).
    // n = group_size = 4 (window_q / window_k grouping marker).
    // k = kv_heads = 2.
    // batch = 1.
    // seq = seq_q = 8.
    // group = head_dim / 16 = 4 (window_size marker).
    KernelKey {
        operator_kind: OperatorKind::Attention,
        attention_kind: Some(AttentionKind::Gqa),
        shape_signature: shape(64, 4, 2, 1, 8, 4),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

#[test]
fn sliding_window_deterministic_picks_lowest_p95_metal_backend() {
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
    let zmm = make_candidate(
        "SlidingWindowZmm",
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
    let key = sliding_window_key();
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
            &samples_with_p95(900),
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
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_metal,
                "Metal p95=900 must beat zmm p95=1300 and scalar p95=4500"
            );
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn sliding_window_trace_lists_chosen_candidate() {
    let min = shape(32, 1, 1, 1, 1, 1);
    let max = shape(128, 8, 8, 4, 256, 8);
    let scalar = make_candidate(
        "SlidingWindowScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "SlidingWindowMetal",
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
    let key = sliding_window_key();
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
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal));
}
