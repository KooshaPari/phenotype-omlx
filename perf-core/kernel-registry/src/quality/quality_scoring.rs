//! Scoring logic: evidence, attachments, errors, and the
//! `evaluate_for_production` convenience builder.
//!
//! Split from `quality.rs` to isolate scoring/evaluation concerns from
//! threshold definitions and promotion audit-trail types.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::record::TuningRecord;

use super::quality_thresholds::QualityGate;

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

    /// Build evidence from an eval-harness [`EvaluationReport`] JSON blob.
    ///
    /// Thin adapter: does **not** depend on the `eval-harness` crate. Expects
    /// the public report shape (`suite`, `accuracy`, …). Gate id is the suite
    /// name with `_` → `-` (e.g. `terminal_bench` → `terminal-bench`).
    ///
    /// Score is `accuracy` (fraction correct). `dataset_revision` is
    /// `eval-harness:<suite>`.
    pub fn from_evaluation_report_json(
        bytes: &[u8],
        source_revision: impl Into<String>,
        captured_at_unix_ms: u64,
    ) -> Result<Self, QualityError> {
        #[derive(Deserialize)]
        struct EvaluationReportView {
            suite: String,
            accuracy: f64,
        }

        let view: EvaluationReportView = serde_json::from_slice(bytes).map_err(|e| {
            QualityError::InvalidEvaluationReport(format!("parse EvaluationReport JSON: {e}"))
        })?;
        if view.suite.trim().is_empty() {
            return Err(QualityError::InvalidEvaluationReport(
                "suite must be a non-empty string".into(),
            ));
        }
        if !view.accuracy.is_finite() {
            return Err(QualityError::InvalidEvaluationReport(
                "accuracy must be a finite f64".into(),
            ));
        }
        let id = view.suite.replace('_', "-");
        Ok(Self {
            id: id.clone(),
            score: view.accuracy,
            dataset_revision: format!("eval-harness:{id}"),
            source_revision: source_revision.into(),
            captured_at_unix_ms,
            note: "imported from EvaluationReport JSON".into(),
        })
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
    #[error("invalid EvaluationReport JSON: {0}")]
    InvalidEvaluationReport(String),
}

/// Lowercase hex encoding of a SHA-256 digest. Stable across builds.
pub fn hex_lower(bytes: &[u8]) -> String {
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
