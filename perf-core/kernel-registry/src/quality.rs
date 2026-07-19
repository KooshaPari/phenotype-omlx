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

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

use crate::candidate::CandidateId;
use crate::record::TuningRecord;

/// One named quality threshold. A gate is *passing* when
/// `evidence.score >= gate.threshold`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityGate {
    /// Stable short id used in rejection traces. Convention: lowercase
    /// hyphenated, e.g. `"mmlu-pro"`, `"gpqa-diamond"`, `"bfcl-ast"`.
    pub id: String,
    /// Threshold the gate enforces. The interpretation depends on
    /// `direction`: for `AtLeast` the score must be `>= threshold`, for
    /// `AtMost` the score must be `<= threshold`.
    pub threshold: f64,
    /// How to interpret `threshold`.
    pub direction: GateDirection,
    /// Optional human-readable note (provenance, dataset revision).
    #[serde(default)]
    pub note: String,
}

impl QualityGate {
    /// Convenience constructor for "score >= threshold".
    pub fn at_least(id: impl Into<String>, threshold: f64) -> Self {
        Self {
            id: id.into(),
            threshold,
            direction: GateDirection::AtLeast,
            note: String::new(),
        }
    }

    /// Convenience constructor for "score <= threshold" (e.g. perplexity).
    pub fn at_most(id: impl Into<String>, threshold: f64) -> Self {
        Self {
            id: id.into(),
            threshold,
            direction: GateDirection::AtMost,
            note: String::new(),
        }
    }

    /// `true` when `score` satisfies the gate under its direction.
    pub fn passes(&self, score: f64) -> bool {
        match self.direction {
            GateDirection::AtLeast => score >= self.threshold,
            GateDirection::AtMost => score <= self.threshold,
        }
    }

    /// Short stable tag for use in rejection strings.
    pub fn tag(&self) -> &str {
        &self.id
    }
}

/// How a [`QualityGate`] compares its score against its threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateDirection {
    /// Higher is better. `score >= threshold` to pass.
    AtLeast,
    /// Lower is better. `score <= threshold` to pass. Used for
    /// perplexity, calibration error, and rejection rates.
    AtMost,
}

/// One named score attached to a tuning record. Carries the dataset
/// revision and source revision so quality regressions are attributable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityEvidence {
    /// Same id as the corresponding [`QualityGate::id`].
    pub id: String,
    /// Observed score on the gate's dataset.
    pub score: f64,
    /// Dataset version (e.g. `"MMLU-Pro@2024-06"`, `"GPQA-Diamond@2024-04"`).
    pub dataset_revision: String,
    /// Source revision that produced this evidence. Should match
    /// [`crate::record::TuningRecord::source_revision`].
    pub source_revision: String,
    /// Capture timestamp in unix-ms.
    pub captured_at_unix_ms: u64,
    /// Optional note (judge model version, evaluator config).
    #[serde(default)]
    pub note: String,
}

impl QualityEvidence {
    /// Construct an evidence entry. `score` is stored verbatim; the gate
    /// decides pass/fail.
    pub fn new(
        id: impl Into<String>,
        score: f64,
        dataset_revision: impl Into<String>,
        source_revision: impl Into<String>,
        captured_at_unix_ms: u64,
    ) -> Self {
        Self {
            id: id.into(),
            score,
            dataset_revision: dataset_revision.into(),
            source_revision: source_revision.into(),
            captured_at_unix_ms,
            note: String::new(),
        }
    }

    /// `true` when this evidence satisfies `gate`.
    pub fn satisfies(&self, gate: &QualityGate) -> bool {
        self.id == gate.id && gate.passes(self.score)
    }
}

/// Optional attachment to a [`crate::record::TuningRecord`] that carries
/// one or more [`QualityEvidence`] entries. Promotion policy consults
/// `gates()` to know which gates must be satisfied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct QualityAttachment {
    /// Every gate whose id appears in the [`QualityEvidence`] entries
    /// must pass for promotion. An empty list means "no quality
    /// requirement" — see [`SelectionPolicy::requires_quality_evidence`].
    #[serde(default)]
    pub gates: Vec<QualityGate>,
    /// Evidence rows. Multiple entries for the same id are not allowed —
    /// the registry treats this as a programming error and rejects the
    /// attachment.
    #[serde(default)]
    pub evidence: Vec<QualityEvidence>,
}

impl QualityAttachment {
    /// Empty attachment — no quality requirement.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Builder-style constructor.
    pub fn new(gates: Vec<QualityGate>, evidence: Vec<QualityEvidence>) -> Self {
        Self { gates, evidence }
    }

    /// `true` when every gate in `self.gates` has a matching evidence row
    /// that passes. Gates with no matching evidence count as a *missing*
    /// requirement (not as a passing one). Duplicate ids in `evidence`
    /// are treated as a programming error.
    pub fn passes_all(&self) -> Result<bool, QualityError> {
        let mut by_id: BTreeMap<&str, &QualityEvidence> = BTreeMap::new();
        for e in &self.evidence {
            if by_id.insert(e.id.as_str(), e).is_some() {
                return Err(QualityError::DuplicateEvidence(e.id.clone()));
            }
        }
        if self.gates.is_empty() {
            return Ok(true);
        }
        for gate in &self.gates {
            match by_id.get(gate.id.as_str()) {
                Some(ev) if gate.passes(ev.score) => continue,
                Some(_) => return Ok(false),
                None => return Ok(false),
            }
        }
        Ok(true)
    }

    /// `true` when the attachment is missing evidence for `gate`.
    pub fn missing_for(&self, gate: &QualityGate) -> bool {
        !self.evidence.iter().any(|e| e.id == gate.id)
    }
}

/// Errors produced by the governance surface.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Serialize, Deserialize)]
pub enum QualityError {
    #[error("duplicate quality evidence for gate id {0}")]
    DuplicateEvidence(String),
    #[error("promotion requires gates, none provided")]
    PromotionWithoutGates,
    #[error("promotion rejected: gate '{gate}' observed={observed:.4} threshold={threshold:.4}")]
    PromotionGateRejected {
        gate: String,
        observed: f64,
        threshold: f64,
    },
    #[error("promotion rejected: gate '{gate}' has no evidence")]
    PromotionGateMissingEvidence { gate: String },
    #[error("promotion rejected: signature mismatch (expected={expected}, got={got})")]
    SignatureMismatch { expected: String, got: String },
}

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
        h.update(&self.canonical_bytes());
        hex_lower(&h.finalize())
    }

    /// Verify that the stored `content_hash` matches the recomputed one.
    /// Returns false when the record was edited after construction.
    pub fn verify_content_hash(&self) -> bool {
        self.content_hash == self.content_hash()
    }

    /// Stable JSON serialization used by `content_hash` and `signature`.
    /// The output is compact and key-ordered to avoid whitespace drift.
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut intermediate = serde_json::Map::new();
        intermediate.insert(
            "candidate_id".to_string(),
            serde_json::to_value(self.candidate_id.0).unwrap_or(serde_json::Value::Null),
        );
        intermediate.insert(
            "source_revision".to_string(),
            serde_json::Value::String(self.source_revision.clone()),
        );
        intermediate.insert(
            "approved_at_unix_ms".to_string(),
            serde_json::to_value(self.approved_at_unix_ms).unwrap_or(serde_json::Value::Null),
        );
        intermediate.insert(
            "approver".to_string(),
            serde_json::Value::String(self.approver.clone()),
        );
        intermediate.insert(
            "gates".to_string(),
            serde_json::to_value(&self.gates).unwrap_or(serde_json::Value::Null),
        );
        intermediate.insert(
            "evidence".to_string(),
            serde_json::to_value(&self.evidence).unwrap_or(serde_json::Value::Null),
        );
        intermediate.insert(
            "justification".to_string(),
            serde_json::Value::String(self.justification.clone()),
        );
        intermediate.insert(
            "tuning_record_id".to_string(),
            match &self.tuning_record_id {
                Some(s) => serde_json::Value::String(s.clone()),
                None => serde_json::Value::Null,
            },
        );
        serde_json::to_vec(&intermediate).unwrap_or_default()
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

/// Lowercase hex encoding of a SHA-256 digest. Stable across builds.
fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

/// Convenience builder: validate a candidate's tuning record + quality
/// attachment against every gate in `policy_gates`. Returns `Ok(())` when
/// the gates pass; otherwise the first failing gate as an error.
pub fn evaluate_for_production(
    record: &TuningRecord,
    attachment: &QualityAttachment,
) -> Result<(), QualityError> {
    if attachment.gates.is_empty() {
        return Err(QualityError::PromotionWithoutGates);
    }
    let mut by_id: BTreeMap<&str, &QualityEvidence> = BTreeMap::new();
    for e in &attachment.evidence {
        if by_id.insert(e.id.as_str(), e).is_some() {
            return Err(QualityError::DuplicateEvidence(e.id.clone()));
        }
    }
    for gate in &attachment.gates {
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
    // Sanity: source revisions must agree.
    for ev in &attachment.evidence {
        if ev.source_revision != record.source_revision {
            return Err(QualityError::PromotionGateRejected {
                gate: ev.id.clone(),
                observed: ev.score,
                threshold: 0.0,
            });
        }
    }
    Ok(())
}