//! Tests for the governance surface introduced alongside the Production
//! selection policy:
//!
//! - `Production` policy refuses candidates without [`kernel_registry::QualityAttachment`].
//! - `Production` refuses candidates whose gates fail.
//! - `PromotionRecord::content_hash` is stable across re-serialization.
//! - `PromotionRecord::verify_signature` round-trips a stable HMAC.
//! - `Metric::EnergyPerOp` and `Metric::Dispatches` select candidates whose
//!   tuning record actually carries the metric.
//! - AtMost gate directions (perplexity, error rate) are respected.

use kernel_registry::{
    evaluate_for_production, BackendKind, Candidate, CandidateId, DeviceCaps, GateDirection,
    KernelKey, KernelRegistry, Metric, PromotionAction, PromotionRecord, PromotionValidator,
    QualityAttachment, QualityError, QualityEvidence, QualityGate, RejectionReason,
    SelectionDecision, SelectionPolicy, ShapeSignature, TuningRecord,
};
use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::record::Measurement;

fn shape(m: usize, n: usize, k: usize) -> ShapeSignature {
    ShapeSignature { m, n, k, batch: 0, seq: 0, group: 0 }
}

fn min_max() -> (ShapeSignature, ShapeSignature) {
    (shape(1, 1, 1), shape(1024, 1024, 1024))
}

fn std_key() -> KernelKey {
    KernelKey {
        operator_kind: OperatorKind::DenseMatmul,
        attention_kind: None,
        shape_signature: ShapeSignature { m: 16, n: 16, k: 16, batch: 0, seq: 0, group: 0 },
        dtype: DType::Fp16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: "test-device".to_string(),
        policy_version: 1,
    }
}

fn make_candidate(
    id_suffix: &str,
    backend: BackendKind,
    (mn, mx): (ShapeSignature, ShapeSignature),
) -> Candidate {
    Candidate {
        id: CandidateId::derive(id_suffix, backend),
        name: id_suffix.to_string(),
        backend,
        source_hash: format!("sha256:{id_suffix}"),
        requires: vec![],
        min_shape: mn,
        max_shape: mx,
        supports_dtypes: vec![DType::Fp16],
        tunable: true,
    }
}

/// Build a record without any quality attachment (used by tests that
/// exercise the "missing evidence" rejection).
fn rec_no_quality(candidate_id: CandidateId, samples: &[u64]) -> TuningRecord {
    let measurements: Vec<Measurement> = samples
        .iter()
        .enumerate()
        .map(|(i, &lat)| Measurement::with_metadata(i as u32, lat, None, None, 0))
        .collect();
    TuningRecord::from_measurements(
        candidate_id,
        std_key(),
        measurements,
        "metal-msl",
        "3.2",
        1_700_000_000_000,
        "rev-test",
        None,
    )
}

/// Build a record with a single MMLU gate that passes.
fn rec_with_quality(candidate_id: CandidateId, samples: &[u64]) -> TuningRecord {
    let measurements: Vec<Measurement> = samples
        .iter()
        .enumerate()
        .map(|(i, &lat)| Measurement::with_metadata(i as u32, lat, None, None, 0))
        .collect();
    let mut r = TuningRecord::from_measurements(
        candidate_id,
        std_key(),
        measurements,
        "metal-msl",
        "3.2",
        1_700_000_000_000,
        "rev-test",
        None,
    );
    let gate = QualityGate::at_least("mmlu-pro", 0.65);
    let evidence = QualityEvidence::new(
        "mmlu-pro",
        0.71,
        "MMLU-Pro@2024-06",
        "rev-test",
        1_700_000_000_000,
    );
    r.quality = Some(QualityAttachment::new(vec![gate], vec![evidence]));
    r
}

fn empty_caps() -> DeviceCaps {
    DeviceCaps::new(vec![])
}

#[test]
fn production_rejects_when_quality_attachment_missing() {
    let cand = make_candidate("a", BackendKind::Metal, min_max());
    let record = rec_no_quality(cand.id, &[100, 110, 120, 130]);
    let mut reg = KernelRegistry::new();
    reg.register_candidate(cand.clone());
    let key = std_key();
    reg.attach_tuning_record(key.clone(), record);

    let policy = SelectionPolicy::Production { gates: Vec::new(), metric: Metric::P95 };
    let decision = reg.select_with_caps(&key, policy, &empty_caps(), 1_700_000_000_000);
    match decision {
        SelectionDecision::Rejected { rejections, .. } => {
            assert!(
                rejections.iter().any(|r| matches!(
                    r.reason,
                    RejectionReason::MissingQualityEvidence(_)
                )),
                "expected MissingQualityEvidence rejection"
            );
        }
        _ => panic!("expected rejection when Production policy needs evidence"),
    }
}

#[test]
fn production_accepts_when_quality_attachment_passes() {
    let cand = make_candidate("a", BackendKind::Metal, min_max());
    let record = rec_with_quality(cand.id, &[100, 110, 120, 130]);
    let mut reg = KernelRegistry::new();
    reg.register_candidate(cand.clone());
    let key = std_key();
    reg.attach_tuning_record(key.clone(), record);

    let policy = SelectionPolicy::Production { gates: Vec::new(), metric: Metric::P95 };
    let decision = reg.select_with_caps(&key, policy, &empty_caps(), 1_700_000_000_000);
    assert!(decision.is_chosen(), "expected chosen with passing evidence");
    assert_eq!(
        decision.selected(),
        Some(cand.id),
        "the only candidate with passing evidence must win"
    );
}

#[test]
fn production_rejects_when_quality_gate_fails() {
    let cand = make_candidate("a", BackendKind::Metal, min_max());
    let measurements: Vec<Measurement> = [100, 110, 120, 130]
        .iter()
        .enumerate()
        .map(|(i, &lat)| Measurement::with_metadata(i as u32, lat, None, None, 0))
        .collect();
    let mut r = TuningRecord::from_measurements(
        cand.id,
        std_key(),
        measurements,
        "metal-msl",
        "3.2",
        1_700_000_000_000,
        "rev-test",
        None,
    );
    let gate = QualityGate::at_least("mmlu-pro", 0.80); // very high bar
    let evidence = QualityEvidence::new(
        "mmlu-pro",
        0.71,
        "MMLU-Pro@2024-06",
        "rev-test",
        1_700_000_000_000,
    );
    r.quality = Some(QualityAttachment::new(vec![gate], vec![evidence]));
    let mut reg = KernelRegistry::new();
    reg.register_candidate(cand.clone());
    let key = std_key();
    reg.attach_tuning_record(key.clone(), r);

    let policy = SelectionPolicy::Production { gates: Vec::new(), metric: Metric::P95 };
    let decision = reg.select_with_caps(&key, policy, &empty_caps(), 1_700_000_000_000);
    match decision {
        SelectionDecision::Rejected { rejections, .. } => {
            assert!(
                rejections.iter().any(|r| matches!(
                    r.reason,
                    RejectionReason::QualityGateFailed { gate: ref g, .. } if g == "mmlu-pro"
                )),
                "expected QualityGateFailed rejection for mmlu-pro"
            );
        }
        _ => panic!("expected rejection when gate threshold not met"),
    }
}

#[test]
fn metric_energy_selects_lower_energy_among_two_candidates() {
    let c1 = make_candidate("low_energy", BackendKind::Metal, min_max());
    let c2 = make_candidate("hi_energy", BackendKind::Metal, min_max());
    let key = std_key();
    let m1: Vec<Measurement> = (0..10)
        .map(|i| Measurement::with_metadata(i, 100, Some(0.5), Some(1), 0))
        .collect();
    let m2: Vec<Measurement> = (0..10)
        .map(|i| Measurement::with_metadata(i, 100, Some(0.9), Some(1), 0))
        .collect();
    let mut r1 = TuningRecord::from_measurements(
        c1.id, key.clone(), m1, "metal-msl", "3.2",
        1_700_000_000_000, "rev-test", None,
    );
    r1.quality = Some(quality_passing());
    let mut r2 = TuningRecord::from_measurements(
        c2.id, key.clone(), m2, "metal-msl", "3.2",
        1_700_000_000_000, "rev-test", None,
    );
    r2.quality = Some(quality_passing());
    let mut reg = KernelRegistry::new();
    reg.register_candidate(c1.clone());
    reg.register_candidate(c2);
    reg.attach_tuning_record(key.clone(), r1);
    reg.attach_tuning_record(key.clone(), r2);
    let policy = SelectionPolicy::Production { gates: Vec::new(), metric: Metric::EnergyPerOp };
    let d = reg.select_with_caps(&key, policy, &empty_caps(), 1_700_000_000_000);
    assert!(d.is_chosen());
    assert_eq!(d.selected(), Some(c1.id));
}

#[test]
fn metric_dispatches_selects_fewer_dispatch_candidate() {
    let c1 = make_candidate("one_dispatch", BackendKind::Metal, min_max());
    let c2 = make_candidate("three_dispatch", BackendKind::Metal, min_max());
    let key = std_key();
    let m1: Vec<Measurement> = (0..10)
        .map(|i| Measurement::with_metadata(i, 100, None, Some(1), 0))
        .collect();
    let m2: Vec<Measurement> = (0..10)
        .map(|i| Measurement::with_metadata(i, 100, None, Some(3), 0))
        .collect();
    let mut r1 = TuningRecord::from_measurements(
        c1.id, key.clone(), m1, "metal-msl", "3.2",
        1_700_000_000_000, "rev-test", None,
    );
    r1.quality = Some(quality_passing());
    let mut r2 = TuningRecord::from_measurements(
        c2.id, key.clone(), m2, "metal-msl", "3.2",
        1_700_000_000_000, "rev-test", None,
    );
    r2.quality = Some(quality_passing());
    let mut reg = KernelRegistry::new();
    reg.register_candidate(c1.clone());
    reg.register_candidate(c2);
    reg.attach_tuning_record(key.clone(), r1);
    reg.attach_tuning_record(key.clone(), r2);
    let policy = SelectionPolicy::Production { gates: Vec::new(), metric: Metric::Dispatches };
    let d = reg.select_with_caps(&key, policy, &empty_caps(), 1_700_000_000_000);
    assert!(d.is_chosen());
    assert_eq!(d.selected(), Some(c1.id));
}

fn quality_passing() -> QualityAttachment {
    let gate = QualityGate::at_least("mmlu-pro", 0.65);
    let evidence = QualityEvidence::new(
        "mmlu-pro",
        0.71,
        "MMLU-Pro@2024-06",
        "rev-test",
        1_700_000_000_000,
    );
    QualityAttachment::new(vec![gate], vec![evidence])
}

#[test]
fn promotion_record_content_hash_is_stable() {
    let rec = PromotionRecord::new(
        CandidateId::derive("a", BackendKind::Metal),
        "rev-2026-07",
        1_700_000_000_000,
        "ci-bot",
        vec![QualityGate::at_least("mmlu-pro", 0.7)],
        vec![QualityEvidence::new(
            "mmlu-pro",
            0.71,
            "MMLU-Pro@2024-06",
            "rev-2026-07",
            1_700_000_000_000,
        )],
        "MMLU-Pro gate holds; p95 within 1.05x.",
        Some("trace-01".into()),
    );
    let h1 = rec.content_hash.clone();
    let json = serde_json::to_string(&rec).expect("serialize");
    let back: PromotionRecord = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(rec.content_hash, h1);
    assert!(back.verify_content_hash());
    let mut bad = back.clone();
    bad.evidence[0].score = 0.99;
    assert!(!bad.verify_content_hash());
}

#[test]
fn promotion_record_signature_round_trip() {
    let mut rec = PromotionRecord::new(
        CandidateId::derive("a", BackendKind::Metal),
        "rev-x",
        1_700_000_000_000,
        "ci-bot",
        vec![QualityGate::at_least("mmlu-pro", 0.7)],
        vec![QualityEvidence::new(
            "mmlu-pro",
            0.71,
            "MMLU-Pro@2024-06",
            "rev-x",
            1_700_000_000_000,
        )],
        "",
        None,
    );
    rec.sign_with(b"signing-key");
    assert!(rec.verify_signature(b"signing-key"));
    assert!(!rec.verify_signature(b"wrong-key"));
}

#[test]
fn evaluate_for_production_requires_gates() {
    let cand = make_candidate("a", BackendKind::Metal, min_max());
    let record = rec_no_quality(cand.id, &[100, 110, 120]);
    let attachment = QualityAttachment::empty();
    assert_eq!(
        evaluate_for_production(&record, &attachment),
        Err(QualityError::PromotionWithoutGates)
    );
}

#[test]
fn gate_direction_at_most_is_respected() {
    let cand = make_candidate("a", BackendKind::Metal, min_max());
    let mut r = rec_no_quality(cand.id, &[100, 110, 120]);
    let gate = QualityGate {
        id: "perplexity".to_string(),
        threshold: 10.0,
        direction: GateDirection::AtMost,
        note: String::new(),
    };
    let evidence = QualityEvidence::new(
        "perplexity",
        8.4,
        "WikiText-103@2024",
        "rev-test",
        1_700_000_000_000,
    );
    r.quality = Some(QualityAttachment::new(vec![gate], vec![evidence]));
    let attachment = r.quality.as_ref().unwrap();
    assert!(evaluate_for_production(&r, attachment).is_ok());
}

#[test]
fn policy_production_metrics_round_trip() {
    let p1 = SelectionPolicy::Production { gates: Vec::new(), metric: Metric::P95 };
    assert_eq!(p1.metric(), Metric::P95);
    let p2 = SelectionPolicy::Production { gates: Vec::new(), metric: Metric::P99 };
    assert_eq!(p2.metric(), Metric::P99);
    let p3 = SelectionPolicy::Production { gates: Vec::new(), metric: Metric::EnergyPerOp };
    assert_eq!(p3.metric(), Metric::EnergyPerOp);
    let p4 = SelectionPolicy::Production { gates: Vec::new(), metric: Metric::Dispatches };
    assert_eq!(p4.metric(), Metric::Dispatches);
    let d = SelectionPolicy::Deterministic { prefer_lower_p95: true };
    assert_eq!(d.metric(), Metric::P95);
    let d99 = SelectionPolicy::Deterministic { prefer_lower_p95: false };
    assert_eq!(d99.metric(), Metric::P99);
    let exp = SelectionPolicy::ExperimentalOnly;
    assert_eq!(exp.metric(), Metric::P95);
}

#[test]
fn metric_extract_handles_missing_data() {
    let cand = make_candidate("a", BackendKind::Metal, min_max());
    let r = rec_no_quality(cand.id, &[100, 110, 120]);
    assert_eq!(Metric::EnergyPerOp.extract(&r), u64::MAX);
    assert_eq!(Metric::Dispatches.extract(&r), u32::MAX as u64);
    assert_eq!(Metric::P95.extract(&r), r.p95_ns);
    assert_eq!(Metric::P99.extract(&r), r.p99_ns);
}

#[test]
fn gate_direction_at_most_rejects_when_above_threshold() {
    let cand = make_candidate("a", BackendKind::Metal, min_max());
    let mut r = rec_no_quality(cand.id, &[100, 110, 120]);
    let gate = QualityGate {
        id: "perplexity".to_string(),
        threshold: 5.0,
        direction: GateDirection::AtMost,
        note: String::new(),
    };
    let evidence = QualityEvidence::new(
        "perplexity",
        8.4,
        "WikiText-103@2024",
        "rev-test",
        1_700_000_000_000,
    );
    r.quality = Some(QualityAttachment::new(vec![gate], vec![evidence]));
    let attachment = r.quality.as_ref().unwrap();
    let result = evaluate_for_production(&r, attachment);
    assert!(matches!(
        result,
        Err(QualityError::PromotionGateRejected { gate: ref g, .. }) if g == "perplexity"
    ));
}

fn passing_record(cand_id: CandidateId) -> PromotionRecord {
    PromotionRecord::new(
        cand_id,
        "rev-test",
        1_700_000_000_000,
        "ci-bot-1",
        vec![QualityGate::at_least("mmlu-pro", 0.65)],
        vec![QualityEvidence::new(
            "mmlu-pro",
            0.71,
            "MMLU-Pro@2024-06",
            "rev-test",
            1_700_000_000_000,
        )],
        "MMLU-Pro gate holds; p95 within 1.05x.",
        Some("trace-01".into()),
    )
}

#[test]
fn promotion_validator_promotes_when_gates_pass() {
    let cand = make_candidate("a", BackendKind::Metal, min_max());
    let record = passing_record(cand.id);
    let v = PromotionValidator { signing_key: None };
    let action = v.promote(record, "ci-bot-1", "auto").expect("promote");
    match action {
        PromotionAction::Promote { record, decision } => {
            assert_eq!(decision, "auto");
            assert!(record.verify_content_hash());
            assert!(record.signature.is_none());
        }
        _ => panic!("expected Promote variant"),
    }
}

#[test]
fn promotion_validator_signs_record_when_key_provided() {
    let cand = make_candidate("a", BackendKind::Metal, min_max());
    let record = passing_record(cand.id);
    let v = PromotionValidator { signing_key: Some(b"k1".to_vec()) };
    let action = v.promote(record, "ci-bot-1", "two-person").expect("promote");
    match action {
        PromotionAction::Promote { record, .. } => {
            assert!(record.signature.is_some());
            assert!(record.verify_signature(b"k1"));
            assert!(!record.verify_signature(b"k2"));
        }
        _ => panic!("expected Promote variant"),
    }
}

#[test]
fn promotion_validator_rejects_when_gate_threshold_unmet() {
    let cand = make_candidate("a", BackendKind::Metal, min_max());
    let record = PromotionRecord::new(
        cand.id,
        "rev-test",
        1_700_000_000_000,
        "ci-bot-1",
        vec![QualityGate::at_least("mmlu-pro", 0.95)], // very high
        vec![QualityEvidence::new(
            "mmlu-pro",
            0.71,
            "MMLU-Pro@2024-06",
            "rev-test",
            1_700_000_000_000,
        )],
        "",
        None,
    );
    let v = PromotionValidator::default();
    assert!(matches!(
        v.promote(record, "ci-bot-1", "auto"),
        Err(QualityError::PromotionGateRejected { gate: ref g, .. }) if g == "mmlu-pro"
    ));
}

#[test]
fn promotion_validator_quarantine_carries_record() {
    let cand = make_candidate("a", BackendKind::Metal, min_max());
    let record = passing_record(cand.id);
    let v = PromotionValidator::default();
    let action = v.quarantine(record, "ci-bot-1", "blocked-on-cve");
    match action {
        PromotionAction::Quarantine { decision, record } => {
            assert_eq!(decision, "blocked-on-cve");
            assert!(record.verify_content_hash());
        }
        _ => panic!("expected Quarantine variant"),
    }
}

#[test]
fn promotion_validator_hold_records_reason() {
    let v = PromotionValidator::default();
    let action = v.hold("awaiting MMLU-Pro score from rev-8");
    match action {
        PromotionAction::Hold { reason } => {
            assert_eq!(reason, "awaiting MMLU-Pro score from rev-8");
        }
        _ => panic!("expected Hold variant"),
    }
}

#[test]
fn promotion_action_serde_round_trip() {
    let cand = make_candidate("a", BackendKind::Metal, min_max());
    let record = passing_record(cand.id);
    let actions = vec![
        PromotionAction::Hold { reason: "awaiting".into() },
        PromotionAction::Quarantine {
            record: record.clone(),
            decision: "blocked".into(),
        },
        PromotionAction::Promote {
            record,
            decision: "auto".into(),
        },
    ];
    for a in actions {
        let json = serde_json::to_string(&a).expect("serialize");
        let back: PromotionAction = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(a, back);
    }
}
