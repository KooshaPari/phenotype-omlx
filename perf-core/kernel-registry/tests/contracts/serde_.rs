//! Serde round-trips for `TuningRecord` and `ExecutionTrace`.

use kernel_registry::compat::AttentionKind;
use kernel_registry::compat::OperatorKind;
use kernel_registry::{CandidateId, ExecutionTrace, KernelKey, TraceRejection, TuningRecord};

use super::{key_with, tuning_record, TEST_DEVICE_FINGERPRINT};

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
    assert!(decoded.considered[0]
        .reason
        .to_lowercase()
        .contains("stale"));
    assert!(decoded.considered[1]
        .reason
        .to_lowercase()
        .contains("metal"));
    assert!(decoded.fallback_used);
    assert_eq!(
        decoded.tuning_record_id.as_deref(),
        Some("tuning-evidence-1")
    );
}
