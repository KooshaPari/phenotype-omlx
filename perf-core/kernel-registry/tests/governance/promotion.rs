//! Promotion validator contract tests.
//!
//! Pins the public contract for `PromotionRecord` (stable content
//! hash, signature round-trip) and `PromotionValidator` (Promote /
//! Hold / Quarantine actions, gate enforcement, optional signing).
//!
//! The `passing_record` helper is local to this module so the file
//! reads top-to-bottom without jumping to `main.rs` for fixtures.
//! Cross-module helpers (`make_candidate`, `min_max`) still come from
//! `super::*`.

use kernel_registry::{
    BackendKind, CandidateId, PromotionAction, PromotionRecord, PromotionValidator, QualityError,
    QualityEvidence, QualityGate,
};

use super::{make_candidate, min_max};

/// Build a `PromotionRecord` whose gates and evidence are coherent and
/// whose threshold (0.65) passes the evidence (0.71).
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
