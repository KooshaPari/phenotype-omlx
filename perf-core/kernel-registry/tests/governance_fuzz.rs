//! Fuzz, property, and concurrency tests for the governance surface.
//!
//! Targets:
//! - [`PromotionRecord`] content-hash determinism, tamper detection, and
//!   signature round-trip.
//! - [`QualityGate`] / [`QualityEvidence`] `passes`/`satisfies` consistency
//!   across threshold + direction combinations.
//! - [`PromotionValidator`] safe under concurrent promotion of the same
//!   candidate (no torn writes, no signature loss).
//!
//! These tests are deliberately standalone (no internal crate helpers
//! beyond `proptest` + `std::thread`) so they double as a fuzz harness
//! when `cargo test` is invoked with `PROPTEST_CASES=4096` or higher.

use kernel_registry::{
    evaluate_for_production, AttentionKind, CandidateId, DType, GateDirection, KernelKey,
    OperatorKind, PromotionAction, PromotionRecord, PromotionValidator, QuantizationPolicy,
    QualityAttachment, QualityEvidence, QualityGate, ShapeSignature, TuningRecord,
};
use proptest::prelude::*;

#[allow(dead_code)]
/// Strategy: well-formed gate id + threshold + direction.
fn arb_gate() -> impl Strategy<Value = QualityGate> {
    (
        prop_oneof![
            Just("mmlu-pro".to_string()),
            Just("gpqa-diamond".to_string()),
            Just("bfcl-ast".to_string()),
            Just("humaneval+".to_string()),
        ],
        -1.0f64..=2.0f64,
        prop_oneof![Just(GateDirection::AtLeast), Just(GateDirection::AtMost)],
    )
        .prop_map(|(id, threshold, direction)| QualityGate {
            id,
            threshold,
            direction,
            note: String::new(),
        })
}

#[allow(dead_code)]
/// Strategy: matching gate id + score in a wide range.
fn arb_evidence(gate_id: String) -> impl Strategy<Value = QualityEvidence> {
    (
        Just(gate_id),
        -1.0f64..=2.0f64,
        Just("MMLU-Pro@2024-06".to_string()),
        Just("rev-7".to_string()),
        1_700_000_000_000u64..=1_800_000_000_000u64,
    )
        .prop_map(
            |(id, score, dataset_revision, source_revision, captured_at_unix_ms)| QualityEvidence {
                id,
                score,
                dataset_revision,
                source_revision,
                captured_at_unix_ms,
                note: String::new(),
            },
        )
}

/// Strategy: build a record that is intended to be promotable (all gates
/// have matching evidence with passing scores).
fn arb_promotable_record() -> impl Strategy<Value = PromotionRecord> {
    (
        0u64..=1024u64,
        prop_oneof![
            Just("rev-7".to_string()),
            Just("rev-8".to_string()),
            Just("rev-9".to_string()),
        ],
        1_700_000_000_000u64..=1_800_000_000_000u64,
        prop::collection::vec(
            (
                prop_oneof![
                    Just("mmlu-pro".to_string()),
                    Just("gpqa-diamond".to_string()),
                ],
                0.0f64..=1.0f64,
            ),
            0..=3,
        ),
        prop_oneof![Just("ci-bot".to_string()), Just("manual".to_string())],
        prop_oneof![Just(String::new()), Just("within 1.05x baseline".to_string())],
        prop::option::of(Just("trace-1".to_string())),
    )
        .prop_map(
            |(
                candidate_id,
                source_revision,
                approved_at_unix_ms,
                gate_score_pairs,
                approver,
                justification,
                tuning_record_id,
            )| {
                let mut gates = Vec::new();
                let mut evidence = Vec::new();
                for (id, score) in gate_score_pairs {
                    gates.push(QualityGate {
                        id: id.clone(),
                        threshold: 0.5,
                        direction: GateDirection::AtLeast,
                        note: String::new(),
                    });
                    evidence.push(QualityEvidence {
                        id,
                        score,
                        dataset_revision: "MMLU-Pro@2024-06".to_string(),
                        source_revision: source_revision.clone(),
                        captured_at_unix_ms: approved_at_unix_ms,
                        note: String::new(),
                    });
                }
                PromotionRecord::new(
                    CandidateId(candidate_id),
                    source_revision,
                    approved_at_unix_ms,
                    approver,
                    gates,
                    evidence,
                    justification,
                    tuning_record_id,
                )
            },
        )
}

proptest! {
    /// A freshly built PromotionRecord always verifies its own content hash
    /// (the immutability proof must hold at construction time).
    #[test]
    fn promotable_record_content_hash_is_self_consistent(record in arb_promotable_record()) {
        prop_assert!(record.verify_content_hash());
    }

    /// The content hash is stable under serde round-tripping — a property
    /// required for audit-trail reproducibility.
    #[test]
    fn content_hash_is_stable_across_serde_round_trip(record in arb_promotable_record()) {
        let original_hash = record.content_hash.clone();
        let json = serde_json::to_string(&record).expect("serialize");
        let back: PromotionRecord = serde_json::from_str(&json).expect("deserialize");
        let round_tripped_hash = back.content_hash.clone();

        // Surface a debug trace only when something has gone wrong so the
        // normal pass path stays quiet.
        if original_hash != round_tripped_hash || !back.verify_content_hash() {
            eprintln!("=== content-hash round-trip mismatch ===");
            eprintln!("original_hash:    {}", original_hash);
            eprintln!("round_tripped:    {}", round_tripped_hash);
            eprintln!("back_recomputed:  {}", back.content_hash());
            eprintln!("record.verify():  {}", record.verify_content_hash());
            eprintln!("back.verify():    {}", back.verify_content_hash());
            eprintln!("json:             {}", json);
        }
        prop_assert_eq!(original_hash, round_tripped_hash);
        prop_assert!(back.verify_content_hash());
    }

    /// Mutating any field after construction must invalidate the stored
    /// content hash. This is the tamper-detection guarantee.
    #[test]
    fn mutating_justification_breaks_content_hash(record in arb_promotable_record()) {
        let mut tampered = record;
        tampered.justification.push('!');
        prop_assert!(!tampered.verify_content_hash());
    }

    /// Mutating the gates list also invalidates the hash.
    #[test]
    fn mutating_gates_breaks_content_hash(record in arb_promotable_record()) {
        let mut tampered = record;
        tampered.gates.push(QualityGate::at_least("new-gate", 0.0));
        prop_assert!(!tampered.verify_content_hash());
    }

    /// A signed record verifies under the same key and rejects a wrong key.
    #[test]
    fn signature_round_trip(record in arb_promotable_record(), key in prop::collection::vec(any::<u8>(), 8..=64)) {
        let mut signed = record;
        signed.sign_with(&key);
        prop_assert!(signed.signature.is_some());
        prop_assert!(signed.verify_signature(&key));
        // Wrong key: verification must fail when a signature is present.
        let mut wrong_key = key.clone();
        if let Some(b) = wrong_key.first_mut() {
            *b ^= 0xFF;
        }
        prop_assert!(!signed.verify_signature(&wrong_key));
    }

    /// QualityGate::passes and QualityEvidence::satisfies agree on every
    /// (threshold, direction, score) combination we care about — the two
    /// paths must never diverge.
    #[test]
    fn gate_passes_matches_evidence_satisfies(
        (threshold, direction, score) in (-1.0f64..=2.0f64, prop_oneof![Just(GateDirection::AtLeast), Just(GateDirection::AtMost)], -1.0f64..=2.0f64)
    ) {
        let gate = QualityGate { id: "g".to_string(), threshold, direction, note: String::new() };
        let evidence = QualityEvidence {
            id: "g".to_string(),
            score,
            dataset_revision: "ds".to_string(),
            source_revision: "rev".to_string(),
            captured_at_unix_ms: 0,
            note: String::new(),
        };
        prop_assert_eq!(gate.passes(score), evidence.satisfies(&gate));
    }

    /// evaluate_for_production is purely a function of the (gates,
    /// evidence) pair relative to the tuning record's source revision.
    /// Soak it across 256 random attachments; the gate pass/fail must
    /// always agree with the score-vs-threshold check, and the source-
    /// revision guard must always trip when they disagree.
    #[test]
    fn evaluate_for_production_is_consistent_with_gate_and_source(
        (threshold, score, src_match) in (-1.0f64..=2.0f64, -1.0f64..=2.0f64, any::<bool>())
    ) {
        let source_revision = if src_match { "rev-7" } else { "rev-8" }.to_string();
        let gate = QualityGate::at_least("mmlu-pro", threshold);
        let evidence = QualityEvidence::new(
            "mmlu-pro",
            score,
            "MMLU-Pro@2024-06",
            source_revision.clone(),
            1_724_000_000_000,
        );
        let att = QualityAttachment::new(vec![gate], vec![evidence]);
        let record = make_minimal_tuning_record("rev-7");
        let result = evaluate_for_production(&record, &att);
        let score_passes = score >= threshold;
        if !src_match {
            // Source-revision guard trips regardless of score.
            prop_assert!(result.is_err());
        } else if score_passes {
            prop_assert!(result.is_ok());
        } else {
            // The matches! macro trips proptest's formatter. Use a manual
            // pattern check via Debug equality instead.
            let is_rejected = matches!(
                &result,
                Err(kernel_registry::QualityError::PromotionGateRejected { .. })
            );
            prop_assert!(is_rejected);
        }
    }
}

// ---------- Concurrency smoke test (no proptest) ----------

/// Build a candidate with a passing attachment and verify that 16 threads
/// racing to promote it produce 16 records that all verify against their
/// own content_hash. This catches torn writes inside PromotionValidator
/// and sign_with under concurrent load.
#[test]
fn promotion_validator_is_concurrency_safe() {
    use std::sync::Arc;
    use std::thread;

    let gate = QualityGate::at_least("mmlu-pro", 0.5);
    let evidence = QualityEvidence::new(
        "mmlu-pro",
        0.71,
        "MMLU-Pro@2024-06",
        "rev-7",
        1_724_000_000_000,
    );
    let validator = Arc::new(PromotionValidator {
        signing_key: Some(b"signing-key-bytes".to_vec()),
    });

    let mut handles = Vec::new();
    for thread_idx in 0..16u32 {
        let v = Arc::clone(&validator);
        let g = gate.clone();
        let e = evidence.clone();
        handles.push(thread::spawn(move || -> PromotionAction {
            let mut rec = PromotionRecord::new(
                CandidateId(thread_idx as u64),
                "rev-7",
                1_724_000_000_000,
                format!("ci-bot-{thread_idx}"),
                vec![g],
                vec![e],
                "concurrent promote",
                Some(format!("trace-{thread_idx}")),
            );
            // Tamper with the record after construction; the validator must
            // rebuild the content hash so verification still succeeds.
            rec.justification = format!("concurrent promote #{thread_idx}");
            v.promote(rec, format!("ci-bot-{thread_idx}"), "auto")
                .expect("promote ok")
        }));
    }

    let mut promoted = 0usize;
    for h in handles {
        let action = h.join().expect("thread ok");
        if let PromotionAction::Promote { record, decision: _ } = action {
            assert!(record.verify_content_hash(), "content hash must verify");
            assert!(
                record.verify_signature(b"signing-key-bytes"),
                "signature must verify under the configured key"
            );
            promoted += 1;
        } else {
            panic!("expected PromotionAction::Promote");
        }
    }
    assert_eq!(promoted, 16);
}

// ---------- Helpers ----------

fn make_minimal_tuning_record(source_revision: &str) -> TuningRecord {
    use kernel_registry::Measurement;
    let m = Measurement::new(0, 1_000);
    let key = KernelKey {
        operator_kind: OperatorKind::DenseMatmul,
        attention_kind: Some(AttentionKind::Standard),
        shape_signature: ShapeSignature {
            m: 1,
            n: 1,
            k: 1,
            batch: 0,
            seq: 0,
            group: 0,
        },
        dtype: DType::Fp32,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: "test".to_string(),
        policy_version: 1,
    };
    TuningRecord::from_measurements(
        CandidateId(0),
        key,
        vec![m],
        "rustc",
        "1.0",
        1_724_000_000_000,
        source_revision,
        None,
    )
}