//! Quality governance: gates, evidence, and promotion records.
//!
//! This module is the runtime side of the governance contract described in
//! `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md`
//! §Quality and Governance:
//!
//! > Selection records and benchmark results are immutable artifacts keyed
//! > by source revision and environment. Quality regressions override
//! > speedups. Experimental kernels remain selectable only through
//! > explicit policy and never become production defaults without
//! > evidence promotion.
//!
//! The three types form a strict DAG:
//!
//! - [`QualityGate`] — a *requirement* (e.g. "MMLU-Pro >= 0.65"). A gate
//!   has no score; it only describes a threshold.
//! - [`QualityEvidence`] — a *measurement* (e.g. "MMLU-Pro = 0.71 at
//!   revision rev-7"). Evidence attaches to a [`crate::record::TuningRecord`]
//!   so the selector can compare it against active gates.
//! - [`PromotionRecord`] — the durable audit-trail artifact produced when
//!   a candidate is approved for production. Promotion is the only path
//!   that moves a candidate from `ExperimentalOnly` to the production
//!   selection surface.
//!
//! ## Hashing and content addressing
//!
//! Promotion records carry a `content_hash` field that is a deterministic
//! SHA-256 over the canonical JSON of every other field. Reviewers can
//! compare the hash against a re-serialization to confirm the record has
//! not been mutated since approval.

#[path = "quality_scoring.rs"]
mod quality_scoring;
#[path = "quality_thresholds.rs"]
mod quality_thresholds;

pub use quality_scoring::{
    evaluate_for_production, hex_lower, QualityAttachment, QualityError, QualityEvidence,
};
pub use quality_thresholds::{GateDirection, QualityGate};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

use crate::candidate::CandidateId;

/// Promotion audit-trail artifact. Immutable once written; the
/// `content_hash` field locks the bytes so a reviewer can detect later
/// edits.
///
/// Promotion is the *only* sanctioned path for a candidate to leave
/// `ExperimentalOnly` and become selectable under
/// [`crate::selector::SelectionPolicy::Production`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotionRecord {
    /// Stable id of the candidate being promoted.
    pub candidate_id: CandidateId,
    /// Source revision under which the promotion was approved.
    pub source_revision: String,
    /// Approval timestamp in unix-ms.
    pub approved_at_unix_ms: u64,
    /// Optional reviewer / CI identity.
    pub approver: String,
    /// The [`QualityGate`]s that must remain passing for as long as this
    /// record is the active promotion. Mirrors the attachment but stored
    /// in the record so the audit trail is self-contained.
    pub gates: Vec<QualityGate>,
    /// Evidence rows captured at approval time.
    pub evidence: Vec<QualityEvidence>,
    /// Optional prose justification ("MMLU-Pro gate holds; p95 within
    /// 1.05x of previous baseline").
    pub justification: String,
    /// Optional tuning record id (`tuning_record_id` from the trace). The
    /// registry stores the matching tuning record under this id.
    pub tuning_record_id: Option<String>,
    /// Optional signed signature (hex SHA-256 over the canonical-JSON of
    /// every other field). `None` for unsigned records; signed records
    /// must round-trip the signature check via
    /// [`PromotionRecord::verify_signature`].
    #[serde(default)]
    pub signature: Option<String>,
    /// Content hash of every other field, computed at construction time
    /// via [`PromotionRecord::content_hash`].
    #[serde(default)]
    pub content_hash: String,
}

impl PromotionRecord {
    /// Build a new unsigned promotion record. The `content_hash` is
    /// computed before the (absent) signature is appended so re-serializing
    /// the record reproduces the same hash.
    pub fn new(
        candidate_id: CandidateId,
        source_revision: impl Into<String>,
        approved_at_unix_ms: u64,
        approver: impl Into<String>,
        gates: Vec<QualityGate>,
        evidence: Vec<QualityEvidence>,
        justification: impl Into<String>,
        tuning_record_id: Option<String>,
    ) -> Self {
        let mut rec = Self {
            candidate_id,
            source_revision: source_revision.into(),
            approved_at_unix_ms,
            approver: approver.into(),
            gates,
            evidence,
            justification: justification.into(),
            tuning_record_id,
            signature: None,
            content_hash: String::new(),
        };
        rec.content_hash = rec.content_hash();
        rec
    }

    /// Sign a record with `signing_key` (any byte string) by storing a
    /// hex SHA-256 of (canonical-JSON of the record, key bytes) as
    /// `signature`. Verification is intentionally symmetric: any caller
    /// with the same key can recompute the signature and compare.
    ///
    /// This is *not* a real cryptographic signature; the contract is that
    /// `signature` is a stable content-keyed MAC suitable for detecting
    /// edits to the record. Promote with HMAC-SHA256 in the consumer when
    /// stronger authenticity is required.
    pub fn sign_with(&mut self, signing_key: &[u8]) {
        let mut buf = self.canonical_bytes();
        buf.extend_from_slice(signing_key);
        let mut h = Sha256::new();
        h.update(&buf);
        self.signature = Some(hex_lower(&h.finalize()));
    }

    /// `true` when the stored signature, if present, matches the
    /// recomputed signature under `signing_key`. A `None` signature is
    /// treated as "unsigned, accept"; a `Some(_)` signature with no
    /// matching key returns false.
    pub fn verify_signature(&self, signing_key: &[u8]) -> bool {
        match &self.signature {
            None => true,
            Some(stored) => {
                let mut buf = self.canonical_bytes();
                buf.extend_from_slice(signing_key);
                let mut h = Sha256::new();
                h.update(&buf);
                &hex_lower(&h.finalize()) == stored
            }
        }
    }

    /// Recompute the content hash from every field *except*
    /// `signature` and `content_hash` itself. Used to detect post-hoc
    /// edits to a promotion record.
    pub fn content_hash(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.canonical_bytes());
        hex_lower(&h.finalize())
    }

    /// Verify that the stored `content_hash` matches the recomputed one.
    /// Returns false when the record was edited after construction.
    pub fn verify_content_hash(&self) -> bool {
        self.content_hash == self.content_hash()
    }

    /// Stable JSON serialization used by `content_hash` and `signature`.
    /// The output is compact, key-ordered, and explicitly sorted so two
    /// equivalent records always hash to the same bytes — even after a
    /// serde round-trip that re-orders fields.
    ///
    /// Robustness notes:
    ///
    /// - The 8 top-level keys are sorted lexicographically and emitted in
    ///   that fixed order so callers cannot observe a different ordering
    ///   across crate versions or between a freshly-built record and a
    ///   round-tripped one.
    /// - `gates` and `evidence` are sorted by their `id` field before
    ///   serialization. The on-the-wire JSON does not promise order, and
    ///   sorting here removes any reliance on Vec insertion order (which
    ///   can drift if a caller mutates the lists in place).
    /// - The Vec elements are serialized via `serde_json::to_string` so
    ///   each `QualityGate` / `QualityEvidence` is encoded using its own
    ///   declared `Serialize` impl — preserving field-name and value
    ///   fidelity — and then concatenated under the sorted top-level key.
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut gates_sorted: Vec<&QualityGate> = self.gates.iter().collect();
        gates_sorted.sort_by(|a, b| a.id.cmp(&b.id));
        let mut evidence_sorted: Vec<&QualityEvidence> = self.evidence.iter().collect();
        evidence_sorted.sort_by(|a, b| a.id.cmp(&b.id));

        let gates_json = serde_json::to_string(&gates_sorted).unwrap_or_else(|_| "[]".to_string());
        let evidence_json =
            serde_json::to_string(&evidence_sorted).unwrap_or_else(|_| "[]".to_string());

        let mut pairs: Vec<(&'static str, String)> = vec![
            (
                "candidate_id",
                serde_json::to_string(&self.candidate_id.0).unwrap_or_else(|_| "0".to_string()),
            ),
            (
                "source_revision",
                serde_json::to_string(&self.source_revision).unwrap_or_default(),
            ),
            (
                "approved_at_unix_ms",
                serde_json::to_string(&self.approved_at_unix_ms)
                    .unwrap_or_else(|_| "0".to_string()),
            ),
            (
                "approver",
                serde_json::to_string(&self.approver).unwrap_or_default(),
            ),
            ("evidence", evidence_json),
            ("gates", gates_json),
            (
                "justification",
                serde_json::to_string(&self.justification).unwrap_or_default(),
            ),
            (
                "tuning_record_id",
                serde_json::to_string(&self.tuning_record_id)
                    .unwrap_or_else(|_| "null".to_string()),
            ),
        ];
        pairs.sort_by(|a, b| a.0.cmp(b.0));
        let mut out = String::with_capacity(256);
        out.push('{');
        for (i, (k, v)) in pairs.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&serde_json::to_string(k).unwrap_or_default());
            out.push(':');
            out.push_str(v);
        }
        out.push('}');
        out.into_bytes()
    }

    /// Validate that every gate in `self.gates` is satisfied by the
    /// supplied evidence. Returns the first failing gate as an error.
    pub fn validate(&self) -> Result<(), QualityError> {
        if self.gates.is_empty() {
            return Err(QualityError::PromotionWithoutGates);
        }
        let mut by_id: BTreeMap<&str, &QualityEvidence> = BTreeMap::new();
        for e in &self.evidence {
            if by_id.insert(e.id.as_str(), e).is_some() {
                return Err(QualityError::DuplicateEvidence(e.id.clone()));
            }
        }
        for gate in &self.gates {
            match by_id.get(gate.id.as_str()) {
                None => {
                    return Err(QualityError::PromotionGateMissingEvidence {
                        gate: gate.id.clone(),
                    })
                }
                Some(ev) if !gate.passes(ev.score) => {
                    return Err(QualityError::PromotionGateRejected {
                        gate: gate.id.clone(),
                        observed: ev.score,
                        threshold: gate.threshold,
                    });
                }
                Some(_) => continue,
            }
        }
        Ok(())
    }
}

/// The action a promotion workflow performs on a candidate.
///
/// The enum is the *audit summary* — concrete enough that a CI report can
/// include it but small enough to stay stable. The textual decision field
/// is intentionally free-form so per-organization workflows can attach
/// policies (`"auto"`, `"manual"`, `"two-person"`) without an enum change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PromotionAction {
    /// Approve a candidate for production. Always followed by a
    /// [`PromotionRecord`] in the same audit-trail event.
    Promote {
        record: PromotionRecord,
        decision: String,
    },
    /// Quarantine a candidate: keep evidence attached but block
    /// production selection. The record inside explains why.
    Quarantine {
        record: PromotionRecord,
        decision: String,
    },
    /// Hold: no promotion decision has been recorded yet (the candidate
    /// is in the queue but awaiting evidence).
    Hold { reason: String },
}

/// Coordinator for the promote/quarantine workflow.
///
/// `PromotionValidator` wraps the [`QualityError`] -> [`PromotionAction`]
/// translation plus content-hashing/sign-hashing. Callers that want to
/// build the same records without the wrapper can use
/// [`PromotionRecord`] + [`evaluate_for_production`] directly.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PromotionValidator {
    /// Optional HMAC-style key used to sign records. `None` leaves
    /// records unsigned (still content-hashed).
    pub signing_key: Option<Vec<u8>>,
}

impl PromotionValidator {
    /// Validate `record` against its gates and produce a
    /// [`PromotionRecord`] with a fresh `content_hash` and an optional
    /// signature derived from `signing_key`. On success returns a
    /// [`PromotionAction::Promote`] carrying the finalized record.
    pub fn promote(
        &self,
        record: PromotionRecord,
        approver: impl Into<String>,
        decision: impl Into<String>,
    ) -> Result<PromotionAction, QualityError> {
        record.validate()?;
        let mut r = record;
        r.approver = approver.into();
        if let Some(k) = &self.signing_key {
            r.sign_with(k);
        }
        r.content_hash = r.content_hash();
        Ok(PromotionAction::Promote {
            record: r,
            decision: decision.into(),
        })
    }

    /// Quarantine a candidate's evidence rather than promote it. The
    /// record stays content-hashed but no signature is appended.
    pub fn quarantine(
        &self,
        record: PromotionRecord,
        approver: impl Into<String>,
        decision: impl Into<String>,
    ) -> PromotionAction {
        let mut r = record;
        r.approver = approver.into();
        r.content_hash = r.content_hash();
        PromotionAction::Quarantine {
            record: r,
            decision: decision.into(),
        }
    }

    /// Build the next "hold" entry for the audit log (caller is waiting
    /// on more evidence). The string is appended verbatim.
    pub fn hold(&self, reason: impl Into<String>) -> PromotionAction {
        PromotionAction::Hold {
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod promotion_hash_tests {
    use super::*;
    use crate::candidate::CandidateId;

    /// Regression: JSON round-trip must not change f64 evidence scores by a
    /// ULP, which would break `verify_content_hash` on deserialized records.
    #[test]
    fn content_hash_survives_serde_round_trip_with_non_shortest_float() {
        let record = PromotionRecord::new(
            CandidateId(0),
            "rev-7",
            1_700_000_000_000,
            "ci-bot",
            vec![QualityGate::at_least("mmlu-pro", 0.5)],
            vec![QualityEvidence::new(
                "mmlu-pro",
                0.21522935540649266,
                "MMLU-Pro@2024-06",
                "rev-7",
                1_700_000_000_000,
            )],
            "",
            None,
        );
        let json = serde_json::to_string(&record).expect("serialize");
        let back: PromotionRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(record.content_hash, back.content_hash);
        assert!(back.verify_content_hash());
    }
}
