//! Selector contract tests: lowest p95, tie-break, expiry, fallback, etc.

use kernel_registry::compat::OperatorKind;
use kernel_registry::selector::{RejectionReason, SelectionDecision};
use kernel_registry::{
    BackendKind, Candidate, CandidateId, Capability, DeviceCaps, KernelKey, KernelRegistry,
    SelectionPolicy,
};

use super::{
    candidate_from, key_with, shape, shape_with, tuning_record, DType, NOW_UNIX_MS,
    TEST_DEVICE_FINGERPRINT,
};

fn setup_registry_with_two_records(
    now: u64,
) -> (KernelRegistry, CandidateId, CandidateId, KernelKey) {
    let mut reg = KernelRegistry::new();
    let c_low = candidate_from("low-p95", BackendKind::Metal, vec![Capability::MetalGpu]);
    let c_high = candidate_from("high-p95", BackendKind::Metal, vec![Capability::MetalGpu]);
    let id_low = c_low.id;
    let id_high = c_high.id;
    reg.register_candidate(c_low.clone());
    reg.register_candidate(c_high.clone());

    let key = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);

    let low_samples: Vec<u64> = (0..20).map(|i| 100 + i).collect();
    let high_samples: Vec<u64> = (0..20).map(|i| 300 + i).collect();

    reg.attach_tuning_record(
        key.clone(),
        tuning_record(
            id_low,
            key.clone(),
            &low_samples,
            Some(now + 86_400_000),
            "clang",
            "17.0.0",
            "rev-abc",
        ),
    );
    reg.attach_tuning_record(
        key.clone(),
        tuning_record(
            id_high,
            key.clone(),
            &high_samples,
            Some(now + 86_400_000),
            "clang",
            "17.0.0",
            "rev-abc",
        ),
    );
    (reg, id_low, id_high, key)
}

#[test]
fn selector_picks_lowest_p95_among_fresh_records() {
    let now = NOW_UNIX_MS;
    let (reg, id_low, id_high, key) = setup_registry_with_two_records(now);
    let caps = DeviceCaps { capabilities: vec![Capability::MetalGpu] };
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &caps,
        now,
    );
    match decision {
        SelectionDecision::Chosen { candidate, tuning } => {
            assert_eq!(candidate.id, id_low,
                "expected lowest-p95 candidate, got high");
            assert_eq!(tuning.candidate_id, id_low);
            // id_high must NOT be the chosen one.
            assert_ne!(candidate.id, id_high);
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn selector_tie_breaks_by_candidate_id_ascending() {
    let now = NOW_UNIX_MS;
    let mut reg = KernelRegistry::new();
    // Two candidates with identical samples (and therefore identical p95).
    let a = candidate_from("alpha-kernel", BackendKind::Metal, vec![Capability::MetalGpu]);
    let b = candidate_from("beta-kernel", BackendKind::Metal, vec![Capability::MetalGpu]);
    let id_a = a.id;
    let id_b = b.id;
    reg.register_candidate(a.clone());
    reg.register_candidate(b.clone());

    let key = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);

    let samples: Vec<u64> = (0..50).map(|i| 200 + (i % 5)).collect();
    reg.attach_tuning_record(
        key.clone(),
        tuning_record(id_a, key.clone(), &samples, Some(now + 86_400_000), "clang", "17.0.0", "rev-x"),
    );
    reg.attach_tuning_record(
        key.clone(),
        tuning_record(id_b, key.clone(), &samples, Some(now + 86_400_000), "clang", "17.0.0", "rev-x"),
    );

    let caps = DeviceCaps { capabilities: vec![Capability::MetalGpu] };
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &caps,
        now,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            // Tie-break: smaller id wins. Order is deterministic regardless
            // of HashMap iteration.
            let expected = id_a.min(id_b);
            assert_eq!(candidate.id, expected,
                "tie-break must select smaller CandidateId; got {candidate:?} expected {expected:?}");
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn selector_rejects_all_when_tuning_records_are_stale() {
    let now = NOW_UNIX_MS;
    let mut reg = KernelRegistry::new();
    let cand = candidate_from("stale-only", BackendKind::Metal, vec![Capability::MetalGpu]);
    let id = cand.id;
    reg.register_candidate(cand);

    let key = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    let samples: Vec<u64> = (0..10).map(|i| 150 + i).collect();
    // expires 1 ms before now
    reg.attach_tuning_record(
        key.clone(),
        tuning_record(id, key.clone(), &samples, Some(now - 1), "clang", "17.0.0", "rev-s"),
    );
    let caps = DeviceCaps { capabilities: vec![Capability::MetalGpu] };
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &caps,
        now,
    );
    match decision {
        SelectionDecision::Rejected { rejections, considered } => {
            assert!(considered.contains(&id));
            assert!(rejections.iter().any(|r| matches!(r.reason, RejectionReason::StaleTuning { .. })),
                "expected StaleTuning reason, got {rejections:?}");
        }
        other => panic!("expected Rejected, got {other:?}"),
    }
}

#[test]
fn selector_falls_back_to_reference_when_no_tuning_records() {
    let now = NOW_UNIX_MS;
    let mut reg = KernelRegistry::new();
    let fast = candidate_from("fast-no-tune", BackendKind::Metal, vec![Capability::MetalGpu]);
    let reference = Candidate {
        id: CandidateId(99),
        name: "reference-cpu".into(),
        backend: BackendKind::Reference,
        source_hash: "ref".into(),
        requires: vec![], // no capability requirements
        min_shape: shape(1, 1, 1),
        max_shape: shape_with(usize::MAX / 4, usize::MAX / 4, usize::MAX / 4, 1, 1, 1),
        supports_dtypes: vec![DType::Fp16],
        tunable: false,
        engine_name: None,
    };
    let id_ref = reference.id;
    let id_fast = fast.id;
    reg.register_candidate(fast);
    reg.register_candidate(reference);

    let key = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    let caps = DeviceCaps { capabilities: vec![Capability::MetalGpu] };
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &caps,
        now,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(candidate.id, id_ref,
                "with no tuning evidence the reference kernel must be chosen");
            assert_ne!(candidate.id, id_fast);
        }
        other => panic!("expected Chosen (reference fallback), got {other:?}"),
    }
    let _ = id_ref;
}

#[test]
fn selector_returns_rejected_with_human_explanation_when_no_candidate_matches() {
    let now = NOW_UNIX_MS;
    let mut reg = KernelRegistry::new();
    let cand = Candidate {
        id: CandidateId(7),
        name: "needs-bf16".into(),
        backend: BackendKind::Metal,
        source_hash: "sha256:needs-bf16".into(),
        requires: vec![Capability::MetalGpu, Capability::Bf16],
        min_shape: shape(1, 1, 1),
        max_shape: shape_with(4096, 4096, 4096, 64, 4096, 64),
        supports_dtypes: vec![DType::Bf16], // does not support Fp16
        tunable: true,
        engine_name: None,
    };
    reg.register_candidate(cand);

    // Pick a key whose dtype the candidate cannot satisfy.
    let mut key = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    key.dtype = DType::Fp16;

    let caps = DeviceCaps { capabilities: vec![Capability::MetalGpu, Capability::Bf16] };
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &caps,
        now,
    );
    let trace = reg.explain(&decision);
    match decision {
        SelectionDecision::Rejected { rejections, considered } => {
            assert!(considered.contains(&CandidateId(7)));
            assert!(rejections.iter().any(|r| matches!(r.reason, RejectionReason::UnsupportedDtype(_))),
                "expected UnsupportedDtype reason, got {rejections:?}");
        }
        other => panic!("expected Rejected, got {other:?}"),
    }
    assert!(!trace.human_explanation.is_empty(),
        "the trace must carry a human-readable explanation even on rejection");
    assert!(trace.human_explanation.to_lowercase().contains("dtype")
        || trace.human_explanation.to_lowercase().contains("reject"),
        "explanation should mention the rejection category: {}", trace.human_explanation);
}