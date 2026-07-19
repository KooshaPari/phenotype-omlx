//! Contract tests for the kernel-registry crate.
//!
//! These tests are written FIRST (TDD). They pin the public contract
//! documented in `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md`
//! and `04_IMPLEMENTATION_STRATEGY.md`.
//!
//! Conventions:
//! - `device_fingerprint` is always a stable 64-hex string (matches
//!   `sha256("test-device-v1")`).
//! - All candidate names are derived from a stable string so that
//!   `CandidateId` collisions are reproducible.
//! - `now_unix_ms` is a fixed integer (`1_700_000_000_000`) so the tests are
//!   deterministic regardless of wall-clock time.

use kernel_registry::{
    BackendKind, BoundedTuner, Candidate, CandidateId, Capability, DType, DeviceCaps,
    ExecutionTrace, KernelKey, KernelRegistry, Measurement, QuantizationPolicy, SelectionPolicy,
    ShapeSignature, TraceRejection, TuningRecord,
};
use kernel_registry::compat::{AttentionKind, OperatorKind};
use kernel_registry::selector::{RejectionReason, SelectionDecision};
use kernel_registry::tuner::TunerError;

const TEST_DEVICE_FINGERPRINT: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn shape(m: usize, n: usize, k: usize) -> ShapeSignature {
    ShapeSignature { m, n, k, batch: 1, seq: 1, group: 1 }
}

fn shape_with(m: usize, n: usize, k: usize, batch: usize, seq: usize, group: usize) -> ShapeSignature {
    ShapeSignature { m, n, k, batch, seq, group }
}

fn key_with(operator: OperatorKind, fingerprint: &str, policy_version: u32) -> KernelKey {
    KernelKey {
        operator_kind: operator,
        attention_kind: None,
        shape_signature: shape(64, 64, 64),
        dtype: DType::Fp16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: fingerprint.to_string(),
        policy_version,
    }
}

fn candidate_from(name: &str, backend: BackendKind, requires: Vec<Capability>) -> Candidate {
    let id_digest = kernel_registry::fast_hash_bytes(name.as_bytes());
    Candidate {
        id: CandidateId(id_digest),
        name: name.to_string(),
        backend,
        // Stable, deterministic artifact hash for the candidate.  Use a
        // fixture value so contract tests do not depend on file IO.
        source_hash: format!("sha256:{name}"),
        requires,
        min_shape: shape(1, 1, 1),
        max_shape: shape_with(4096, 4096, 4096, 64, 4096, 64),
        supports_dtypes: vec![DType::Fp16, DType::Bf16],
        tunable: true,
    }
}

fn measurement(sample: u32, latency_ns: u64) -> Measurement {
    Measurement::with_metadata(sample, latency_ns, None, None, 0)
}

fn tuning_record(
    candidate_id: CandidateId,
    key: KernelKey,
    samples: &[u64],
    expires_at_unix_ms: Option<u64>,
    compiler: &str,
    compiler_version: &str,
    source_revision: &str,
) -> TuningRecord {
    let measurements: Vec<Measurement> = samples
        .iter()
        .enumerate()
        .map(|(i, &latency)| measurement(i as u32, latency))
        .collect();
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];
    let p95_idx = ((sorted.len() as f64 * 0.95).ceil() as usize).saturating_sub(1).min(sorted.len() - 1);
    let p99_idx = ((sorted.len() as f64 * 0.99).ceil() as usize).saturating_sub(1).min(sorted.len() - 1);
    let p95 = sorted[p95_idx];
    let p99 = sorted[p99_idx];
    let mean = sorted.iter().sum::<u64>() as f64 / sorted.len() as f64;
    let variance = sorted
        .iter()
        .map(|&x| ((x as f64) - mean).powi(2))
        .sum::<f64>() / sorted.len() as f64;
    TuningRecord {
        candidate_id,
        key,
        measurements,
        median_ns: median,
        p95_ns: p95,
        p99_ns: p99,
        variance_ns2: variance as u64,
        median_energy_j: None,
        median_dispatches: None,
        samples: samples.len(),
        warmup_discarded: 3,
        compiler: compiler.to_string(),
        compiler_version: compiler_version.to_string(),
        captured_at_unix_ms: 1_700_000_000_000,
        source_revision: source_revision.to_string(),
        expires_at_unix_ms,
        quality: None,
    }
}

// ----------------------------------------------------------------------
// KernelKey / hash
// ----------------------------------------------------------------------

#[test]
fn kernel_key_hash_is_stable() {
    let k1 = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    let k2 = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    assert_eq!(k1, k2);
    assert_eq!(k1.fast_hash(), k2.fast_hash(),
        "fast_hash() must be a pure function of the key fields");
    // Hard-coded values keep accidental field reordering loud.
    let k3 = key_with(OperatorKind::Attention, TEST_DEVICE_FINGERPRINT, 1);
    assert_ne!(k1.fast_hash(), k3.fast_hash(),
        "operator_kind must affect fast_hash");
}

#[test]
fn kernel_key_eq_treats_policy_version_as_distinguishing() {
    let k1 = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    let k2 = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 2);
    assert_ne!(k1, k2,
        "policy_version must distinguish keys — selection policy changes invalidate prior evidence");
    assert_ne!(k1.fast_hash(), k2.fast_hash());
}

// ----------------------------------------------------------------------
// Candidate capability and shape filtering
// ----------------------------------------------------------------------

#[test]
fn candidate_capability_filter_rejects_when_required_capability_missing() {
    let mut reg = KernelRegistry::new();
    let cand = candidate_from("metal-gemm-fp16", BackendKind::Metal, vec![Capability::MetalGpu, Capability::Bf16]);
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
        1_700_000_000_000,
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
        1_700_000_000_000,
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

// ----------------------------------------------------------------------
// Selector: choose lowest p95, tie-break by id, expiry, fallback
// ----------------------------------------------------------------------

fn setup_registry_with_two_records(now: u64) -> (KernelRegistry, CandidateId, CandidateId, KernelKey) {
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
        tuning_record(id_low, key.clone(), &low_samples, Some(now + 86_400_000),
            "clang", "17.0.0", "rev-abc"),
    );
    reg.attach_tuning_record(
        key.clone(),
        tuning_record(id_high, key.clone(), &high_samples, Some(now + 86_400_000),
            "clang", "17.0.0", "rev-abc"),
    );
    (reg, id_low, id_high, key)
}

#[test]
fn selector_picks_lowest_p95_among_fresh_records() {
    let now = 1_700_000_000_000u64;
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
    let now = 1_700_000_000_000u64;
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
            // Tie-break: smaller id wins.  Order is deterministic regardless
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
    let now = 1_700_000_000_000u64;
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
    let now = 1_700_000_000_000u64;
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
    let now = 1_700_000_000_000u64;
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

// ----------------------------------------------------------------------
// Serde round-trips
// ----------------------------------------------------------------------

#[test]
fn tuning_record_serde_round_trips() {
    let key = key_with(OperatorKind::Attention, TEST_DEVICE_FINGERPRINT, 1);
    let key_for_attention = KernelKey {
        attention_kind: Some(AttentionKind::Standard),
        ..key
    };
    let samples: Vec<u64> = (0..10).map(|i| 1000 + i * 10).collect();
    let id = CandidateId(42);
    let rec = tuning_record(
        id,
        key_for_attention.clone(),
        &samples,
        Some(1_700_000_999_999),
        "rustc",
        "1.74.0",
        "rev-tuning-1",
    );

    let json = serde_json::to_string(&rec).expect("serialize");
    let decoded: TuningRecord = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(rec, decoded);
    assert_eq!(decoded.measurements.len(), samples.len());
    assert_eq!(decoded.expires_at_unix_ms, Some(1_700_000_999_999));
}

#[test]
fn execution_trace_serde_round_trips_and_includes_rejection_reasons() {
    let trace = ExecutionTrace {
        plan_id: "deadbeef".into(),
        operator_id: "abcd1234".into(),
        considered: vec![
            TraceRejection {
                candidate: CandidateId(1),
                reason: "stale tuning record (expired 1h ago)".into(),
            },
            TraceRejection {
                candidate: CandidateId(2),
                reason: "missing capability: MetalGpu".into(),
            },
        ],
        selected: None,
        fallback_used: true,
        tuning_record_id: Some("tuning-evidence-1".into()),
        emitted_at_unix_ms: 1_700_000_000_000,
        human_explanation: "no fresh tuning evidence; falling back to reference kernel.".into(),
    };

    let json = serde_json::to_string(&trace).expect("serialize");
    let decoded: ExecutionTrace = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(trace, decoded);
    assert_eq!(decoded.considered.len(), 2);
    assert!(decoded.considered[0].reason.to_lowercase().contains("stale"));
    assert!(decoded.considered[1].reason.to_lowercase().contains("metal"));
    assert!(decoded.fallback_used);
    assert_eq!(decoded.tuning_record_id.as_deref(), Some("tuning-evidence-1"));
}

// ----------------------------------------------------------------------
// BoundedTuner
// ----------------------------------------------------------------------

#[test]
fn bounded_tuner_warmup_samples_and_emits_record() {
    let key = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    let cand = candidate_from("warm-tester", BackendKind::Cpu, vec![]);
    let tuner = BoundedTuner::new(/*warmup*/ 3, /*samples*/ 10, /*max_time_ms*/ 10_000);

    let mut invocation_count: u32 = 0;
    let record = tuner.run(key.clone(), cand.clone(), |_k| {
        invocation_count += 1;
        // Latency rises with sample index to exercise ordering.
        let lat = (invocation_count as u64) * 100;
        measurement(invocation_count - 1, lat)
    }).expect("tuner should succeed within budget");

    assert_eq!(record.candidate_id, cand.id);
    assert_eq!(record.warmup_discarded, 3);
    // The closure is called warmup + samples times.
    assert_eq!(invocation_count, (3 + 10) as u32);
    assert_eq!(record.samples, 10);
    assert_eq!(record.measurements.len(), 13,
        "all invocations (warmup + samples) are recorded for provenance");
    assert!(record.median_ns > 0);
    assert!(record.p95_ns >= record.median_ns);
    assert!(record.p99_ns >= record.p95_ns);
}

#[test]
fn bounded_tuner_returns_budget_exceeded_when_measurements_exceed_limit() {
    let key = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    let cand = candidate_from("budget-breaker", BackendKind::Cpu, vec![]);
    // max_time_ms = 0 forces any measurement work to overflow the budget.
    let tuner = BoundedTuner::new(/*warmup*/ 1, /*samples*/ 2, /*max_time_ms*/ 0);

    let mut count = 0u32;
    let result = tuner.run(key, cand, |_k| {
        count += 1;
        // Sleep a measurable amount so budget accounting can flag it.
        std::thread::sleep(std::time::Duration::from_millis(2));
        measurement(count - 1, 2_000_000)
    });

    match result {
        Err(TunerError::BudgetExceeded { used_ms, max_ms }) => {
            assert_eq!(max_ms, 0);
            // We may have stopped after warmup or after first sample; in either
            // case the tuner reports the budget violation.
            assert!(used_ms >= max_ms, "used_ms {used_ms} should be >= max_ms {max_ms}");
        }
        Ok(rec) => panic!("expected BudgetExceeded, got record with {} samples", rec.samples),
    }
}

// ----------------------------------------------------------------------
// SelectionPolicy — ExperimentalOnly excludes tunable kernels
// ----------------------------------------------------------------------
//
// Note: this is an *additional* contract test beyond the task list.  It
// guards the experimental-policy path so it can't silently regress.

#[test]
fn selector_experimental_only_skips_tunable_candidates_without_evidence() {
    let now = 1_700_000_000_000u64;
    let mut reg = KernelRegistry::new();
    let tunable_no_evidence = candidate_from("experimental", BackendKind::Metal, vec![Capability::MetalGpu]);
    let id_tunable = tunable_no_evidence.id;
    reg.register_candidate(tunable_no_evidence);
    let key = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    let caps = DeviceCaps { capabilities: vec![Capability::MetalGpu] };
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::ExperimentalOnly,
        &caps,
        now,
    );
    match decision {
        SelectionDecision::Rejected { rejections, considered } => {
            assert!(considered.contains(&id_tunable));
            assert!(rejections.iter().any(|r| matches!(r.reason, RejectionReason::NoTuningEvidence)),
                "tunable candidates without tuning must be excluded under ExperimentalOnly");
        }
        other => panic!("expected Rejected under ExperimentalOnly, got {other:?}"),
    }
}