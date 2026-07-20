//! (j) MoD sparse depth routing — OperatorKind::Moe, capacity_factor ∈ (0,1]
//!
//! MoD (Mixture-of-Depths) shares the routing discriminator with MoE —
//! shape carries `seq` = full-token count, `batch` = mean capacity target.

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::{
    CandidateId, KernelKey, QualityAttachment, QualityEvidence, QualityGate, TuningRecord,
};

use super::{
    build_record, NOW_UNIX_MS, TEST_FINGERPRINT,
};

pub(super) fn mod_key() -> KernelKey {
    // m = dim (hidden size), seq = full-token count (32), batch = 1.
    KernelKey {
        operator_kind: OperatorKind::Moe,
        attention_kind: None,
        shape_signature: super::shape(8, 8, 8, 1, 32, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

/// Build a tuning record whose `p95_ns` is exactly `p95` and whose quality
/// attachment satisfies the supplied gate id with score `evidence_score`.
pub(super) fn build_mod_record(
    candidate_id: CandidateId,
    key: KernelKey,
    samples: &[u64],
    expires_at_unix_ms: Option<u64>,
    gate_id: &str,
    threshold: f64,
    evidence_score: f64,
) -> TuningRecord {
    let mut r = build_record(candidate_id, key, samples, expires_at_unix_ms);
    let gate = QualityGate::at_least(gate_id, threshold);
    // Evidence `source_revision` MUST match the tuning record's
    // (`quality::evaluate_for_production`); build_record uses `"rev-sota"`.
    let evidence = QualityEvidence::new(
        gate_id,
        evidence_score,
        "ModRoutingPerf@2026-07",
        r.source_revision.clone(),
        NOW_UNIX_MS,
    );
    r.quality = Some(QualityAttachment::new(vec![gate], vec![evidence]));
    r
}

mod policy;