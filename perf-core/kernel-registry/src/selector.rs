//! Selection policy and decision types.
//!
//! The selector is the *only* component in the registry that decides which
//! kernel runs. Its two responsibilities are:
//!
//! 1. Filter candidates by capability, shape, dtype, and freshness of
//!    evidence, accumulating a [`RejectionReason`] for each.
//! 2. Pick the surviving candidates deterministically — first by the
//!    policy metric (`p95_ns` for `Deterministic`, lower-is-better), then
//!    by [`CandidateId`] ascending so the result is reproducible.
//!
//! `ExperimentalOnly` is reserved for offline promotion experiments; it
//! only considers candidates that already carry tuning evidence, so an
//! `ExperimentalOnly` selection that returns a candidate is guaranteed to
//! have fresh evidence attached.

use serde::{Deserialize, Serialize};

use crate::candidate::{Candidate, CandidateId};
use crate::record::TuningRecord;

/// Policy that decides which surviving candidate wins.
///
/// `Deterministic { prefer_lower_p95 }` orders candidates by `p95_ns`
/// ascending when `prefer_lower_p95` is `true`, otherwise by `p99_ns`
/// ascending. The metric is followed by `candidate_id` ascending so the
/// outcome is reproducible regardless of HashMap iteration order.
///
/// `ExperimentalOnly` is the offline promotion policy: it only considers
/// candidates with a non-stale tuning record and only candidates with
/// `tunable == true`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionPolicy {
    Deterministic { prefer_lower_p95: bool },
    ExperimentalOnly,
}

impl SelectionPolicy {
    /// Metric the policy uses to rank candidates.
    pub fn metric(&self) -> Metric {
        match self {
            SelectionPolicy::Deterministic { prefer_lower_p95 } => {
                if *prefer_lower_p95 {
                    Metric::P95
                } else {
                    Metric::P99
                }
            }
            SelectionPolicy::ExperimentalOnly => Metric::P95,
        }
    }
}

/// The latency metric used by a [`SelectionPolicy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    P95,
    P99,
}

impl Metric {
    pub fn extract(self, rec: &TuningRecord) -> u64 {
        match self {
            Metric::P95 => rec.p95_ns,
            Metric::P99 => rec.p99_ns,
        }
    }
}

/// Why the selector rejected a candidate. The `candidate` field carries the
/// id so traces and explanations are unambiguous when several candidates
/// share a rejection category.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectionRecord {
    pub candidate: CandidateId,
    pub reason: RejectionReason,
}

/// The reason category for a [`RejectionRecord`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectionReason {
    /// Required capability is missing on the device.
    MissingCapability(String),
    /// Required dtype is not in `supports_dtypes`.
    UnsupportedDtype(String),
    /// Key shape is outside `[min_shape, max_shape]`.
    ShapeOutOfRange,
    /// A tuning record exists but has expired.
    StaleTuning {
        expires_at_unix_ms: u64,
        now_unix_ms: u64,
    },
    /// No tuning evidence present for this key.
    NoTuningEvidence,
    /// Backend is not selectable under the active policy.
    PolicyExcluded(String),
    /// Catch-all.
    Other(String),
}

impl RejectionReason {
    /// Human-readable form, used in [`crate::ExecutionTrace`].
    pub fn human(&self) -> String {
        match self {
            RejectionReason::MissingCapability(c) => {
                format!("missing capability: {}", c)
            }
            RejectionReason::UnsupportedDtype(d) => {
                format!("unsupported dtype: {:?}", d)
            }
            RejectionReason::ShapeOutOfRange => "shape out of range".to_string(),
            RejectionReason::StaleTuning { expires_at_unix_ms, now_unix_ms } => format!(
                "stale tuning record (expired at {}, now {})",
                expires_at_unix_ms, now_unix_ms
            ),
            RejectionReason::NoTuningEvidence => "no tuning evidence".to_string(),
            RejectionReason::PolicyExcluded(p) => format!("policy excluded: {}", p),
            RejectionReason::Other(o) => o.clone(),
        }
    }
}

impl RejectionRecord {
    pub fn new(candidate: CandidateId, reason: RejectionReason) -> Self {
        Self { candidate, reason }
    }
}

/// The outcome of a single [`crate::KernelRegistry::select_with_caps`] call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SelectionDecision {
    Chosen {
        candidate: Candidate,
        tuning: TuningRecord,
    },
    Rejected {
        rejections: Vec<RejectionRecord>,
        considered: Vec<CandidateId>,
    },
}

impl SelectionDecision {
    /// `true` when the selector made a choice.
    pub fn is_chosen(&self) -> bool {
        matches!(self, SelectionDecision::Chosen { .. })
    }

    /// The chosen candidate id, if any.
    pub fn selected(&self) -> Option<CandidateId> {
        match self {
            SelectionDecision::Chosen { candidate, .. } => Some(candidate.id),
            SelectionDecision::Rejected { .. } => None,
        }
    }
}