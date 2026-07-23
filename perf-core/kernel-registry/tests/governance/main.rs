//! Tests for the governance surface introduced alongside the Production
//! selection policy:
//!
//! - `Production` policy refuses candidates without [`kernel_registry::QualityAttachment`].
//! - `Production` refuses candidates whose gates fail.
//! - `Metric::EnergyPerOp` and `Metric::Dispatches` select candidates whose
//!   tuning record actually carries the metric.
//! - AtMost gate directions (perplexity, error rate) are respected.
//!
//! Helpers in this file (`shape`, `std_key`, `make_candidate`,
//! `rec_no_quality`, `rec_with_quality`, `quality_passing`, `empty_caps`)
//! are the shared foundation for `quality` and `promotion` tests below.
//! Promotion-record and Promotion-validator contract tests live in
//! `promotion.rs`; Quality-gate and metric tests live in `quality.rs`.

use kernel_registry::{
    BackendKind, Candidate, CandidateId, DeviceCaps, KernelKey, KernelRegistry, Metric,
    QualityAttachment, QualityEvidence, QualityGate, RejectionReason, SelectionDecision,
    SelectionPolicy, ShapeSignature, TuningRecord,
};
use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::record::Measurement;

fn shape(m: usize, n: usize, k: usize) -> ShapeSignature {
    ShapeSignature { m, n, k, batch: 0, seq: 0, group: 0 }
}

fn min_max() -> (ShapeSignature, ShapeSignature) {
    (shape(1, 1, 1), shape(1024, 1024, 1024))
}

pub(crate) fn std_key() -> KernelKey {
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
        engine_name: None,
        properties: std::collections::HashMap::new(),
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

mod promotion;
mod quality;
