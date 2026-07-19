//! Structured observability for every selection decision.
//!
//! `kernel-registry` emits an [`ExecutionTrace`] for every call to
//! [`crate::KernelRegistry::select_with_caps`]. The trace is the durable
//! evidence that explains *why* a particular candidate was chosen or
//! rejected and is the input to quality-gate promotion.

use serde::{Deserialize, Serialize};

use crate::candidate::CandidateId;
use crate::selector::SelectionDecision;

/// A single rejection reason for one considered candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceRejection {
    pub candidate: CandidateId,
    pub reason: String,
}

/// The structured trace emitted on every selection.
///
/// `plan_id` is the hex of the [`crate::compat::OperatorKind`]-derived plan
/// identifier (e.g. a `ModelId`). `operator_id` is the hex of the
/// `OperatorId` for the operator being scheduled. `considered` enumerates
/// every candidate the selector examined, paired with a human-readable
/// rejection reason. `selected` is `Some` when the selector chose a
/// candidate (including the reference fallback). `fallback_used` flags
/// whether the selector had to drop back to the reference or a non-optimal
/// candidate because no fresh evidence was available. `tuning_record_id`,
/// when present, is the stable id of the [`crate::TuningRecord`] that
/// justified the selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionTrace {
    pub plan_id: String,
    pub operator_id: String,
    pub considered: Vec<TraceRejection>,
    pub selected: Option<CandidateId>,
    pub fallback_used: bool,
    pub tuning_record_id: Option<String>,
    pub emitted_at_unix_ms: u64,
    pub human_explanation: String,
}

impl ExecutionTrace {
    /// Build a trace from a [`SelectionDecision`]. This is the canonical
    /// entry point used by [`crate::KernelRegistry::explain`]. The
    /// `plan_id`, `operator_id`, and `emitted_at_unix_ms` fields are
    /// filled by the caller when the trace is wired into a real
    /// `ModelPlan`/`OperatorPlan` pipeline; here we leave them empty
    /// because the registry does not know about plan identity.
    pub fn from_decision(decision: &SelectionDecision) -> Self {
        match decision {
            SelectionDecision::Chosen { candidate, tuning } => {
                let considered: Vec<TraceRejection> = Vec::new();
                let fallback_used = candidate.backend.is_reference();
                let tuning_record_id = Some(format!(
                    "{:016x}",
                    tuning.candidate_id.0 ^ 0x9e3779b97f4a7c15
                ));
                Self::build(
                    String::new(),
                    String::new(),
                    considered,
                    Some(candidate.id),
                    fallback_used,
                    tuning_record_id,
                    0,
                )
            }
            SelectionDecision::Rejected { rejections, considered: _ } => {
                let mut sorted: Vec<&crate::selector::RejectionRecord> = rejections.iter().collect();
                sorted.sort_by_key(|r| r.candidate);
                let considered: Vec<TraceRejection> = sorted
                    .into_iter()
                    .map(|r| TraceRejection {
                        candidate: r.candidate,
                        reason: r.reason.human(),
                    })
                    .collect();
                Self::build(
                    String::new(),
                    String::new(),
                    considered,
                    None,
                    false,
                    None,
                    0,
                )
            }
        }
    }

    /// Build a trace from a list of (candidate, reason) pairs and an
    /// optional selection. The human explanation is constructed by
    /// [`ExecutionTrace::format_explanation`].
    pub fn build(
        plan_id: impl Into<String>,
        operator_id: impl Into<String>,
        considered: Vec<TraceRejection>,
        selected: Option<CandidateId>,
        fallback_used: bool,
        tuning_record_id: Option<String>,
        emitted_at_unix_ms: u64,
    ) -> Self {
        let human_explanation = Self::format_explanation(
            &considered,
            selected,
            fallback_used,
            tuning_record_id.as_deref(),
        );
        Self {
            plan_id: plan_id.into(),
            operator_id: operator_id.into(),
            considered,
            selected,
            fallback_used,
            tuning_record_id,
            emitted_at_unix_ms,
            human_explanation,
        }
    }

    /// Compose a stable, human-readable summary of the selection.
    ///
    /// The format is intentionally simple and parseable:
    ///
    /// - `selected <id> via <tuning_record_id>` when a tuned choice exists;
    /// - `selected <id> as reference fallback` when the reference kernel
    ///   was chosen because no fresh evidence was available;
    /// - `rejected: <n> considered, top reasons: <reason1>; <reason2>; ...`
    ///   when the selector rejected all candidates.
    pub fn format_explanation(
        considered: &[TraceRejection],
        selected: Option<CandidateId>,
        fallback_used: bool,
        tuning_record_id: Option<&str>,
    ) -> String {
        if let Some(sel) = selected {
            if let Some(tr) = tuning_record_id {
                return format!("selected {} via tuning record {}", sel, tr);
            }
            if fallback_used {
                return format!("selected {} as reference fallback", sel);
            }
            return format!("selected {} (no tuning evidence required)", sel);
        }
        // Rejected path.
        let mut counts: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for r in considered {
            // Drop the candidate-specific prefix from the reason so we can
            // summarize categories ("missing capability", "stale tuning",
            // "shape out of range", ...).
            let category = categorize_reason(&r.reason);
            *counts.entry(category).or_insert(0) += 1;
        }
        let mut top: Vec<(String, usize)> = counts.into_iter().collect();
        top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let top_str = top
            .into_iter()
            .take(3)
            .map(|(cat, n)| format!("{} ({})", cat, n))
            .collect::<Vec<_>>()
            .join("; ");
        format!(
            "rejected: {} considered; top reasons: {}",
            considered.len(),
            if top_str.is_empty() { "none".to_string() } else { top_str }
        )
    }
}

fn categorize_reason(reason: &str) -> String {
    let lower = reason.to_lowercase();
    if lower.contains("capability") {
        "missing capability".to_string()
    } else if lower.contains("stale") {
        "stale tuning".to_string()
    } else if lower.contains("shape") || lower.contains("range") {
        "shape out of range".to_string()
    } else if lower.contains("dtype") {
        "unsupported dtype".to_string()
    } else if lower.contains("policy") {
        "policy excluded".to_string()
    } else if lower.contains("no tuning") || lower.contains("no evidence") {
        "no tuning evidence".to_string()
    } else {
        "other".to_string()
    }
}