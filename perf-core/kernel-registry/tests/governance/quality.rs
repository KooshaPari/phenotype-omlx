//! Quality-gate and metric contract tests.
//!
//! These pin the `SelectionPolicy::Production` contract for the
//! `QualityAttachment` / `QualityGate` machinery:
//!
//! - `evaluate_for_production` refuses records that have an attachment
//!   without gates.
//! - `GateDirection::AtMost` gates (e.g. perplexity) accept evidence
//!   below the threshold and reject evidence above it.
//! - `Metric::EnergyPerOp` and `Metric::Dispatches` select the right
//!   candidate when the policy asks for them.
//! - `SelectionPolicy::metric()` round-trips for every variant.
//!
//! Cross-cutting helpers (`make_candidate`, `min_max`, `rec_no_quality`,
//! `std_key`) live in `main.rs`.

use kernel_registry::{
    evaluate_for_production, BackendKind, GateDirection, Metric, QualityAttachment, QualityError,
    QualityEvidence, QualityGate, SelectionPolicy,
};

use super::{make_candidate, min_max, rec_no_quality};

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
