//! Selection policy and decision types.
//!
//! The selector is the *only* component in the registry that decides which
//! kernel runs. Its three responsibilities are:
//!
//! 1. Filter candidates by capability, shape, dtype, freshness of
//!    evidence, and (under [`SelectionPolicy::Production`]) quality
//!    evidence; accumulate a [`RejectionReason`] for each rejection.
//! 2. Pick the surviving candidates deterministically — first by the
//!    policy metric (`p95_ns` for `Deterministic`, lower-is-better), then
//!    by [`CandidateId`] ascending so the result is reproducible.
//! 3. Under `Production`, refuse to promote a candidate whose
//!    [`crate::record::TuningRecord`] does not carry an attached quality
//!    attestation that satisfies every active [`crate::quality::QualityGate`].
//!
//! `ExperimentalOnly` is reserved for offline promotion experiments; it
//! only considers candidates that already carry tuning evidence, so an
//! `ExperimentalOnly` selection that returns a candidate is guaranteed to
//! have fresh evidence attached.

use serde::{Deserialize, Serialize};

use crate::candidate::{Candidate, CandidateId};
use crate::quality::QualityGate;
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
///
/// `Production { gates, metric }` is the only policy that allows a
/// candidate to run in production. It refuses to select any candidate
/// whose [`crate::record::TuningRecord`] does not satisfy every
/// [`QualityGate`] in `gates`. The selector ranking metric is governed
/// by `metric` and defaults to `Metric::P95`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SelectionPolicy {
    Deterministic { prefer_lower_p95: bool },
    ExperimentalOnly,
    Production {
        gates: Vec<QualityGate>,
        metric: Metric,
    },
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
            SelectionPolicy::Production { metric, .. } => *metric,
        }
    }

    /// Quality gates that [`SelectionPolicy::Production`] enforces. Other
    /// variants return an empty slice because quality enforcement is only
    /// meaningful for production selection.
    pub fn gates(&self) -> &[QualityGate] {
        match self {
            SelectionPolicy::Production { gates, .. } => gates.as_slice(),
            _ => &[],
        }
    }

    /// `true` when the policy requires fresh tuning evidence attached to a
    /// [`crate::quality::QualityEvidence`]. `Production` is the only policy
    /// that does; `Deterministic` and `ExperimentalOnly` may run on
    /// performance-only evidence.
    pub fn requires_quality_evidence(&self) -> bool {
        matches!(self, SelectionPolicy::Production { .. })
    }
}

/// The latency / energy / dispatch metric used by a [`SelectionPolicy`].
///
/// `EnergyPerOp` and `Dispatches` only rank candidates that have measured
/// those axes; candidates missing the metric fall through to the lower
/// `Metric` variant so the selector still makes progress. The
/// [`SelectionPolicy::Production`] policy additionally requires fresh
/// quality evidence, so a candidate missing energy or dispatch data can
/// still be promoted as long as the gates pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    P95,
    P99,
    /// Joules consumed per invocation (lower-is-better).
    EnergyPerOp,
    /// Dispatch count per invocation (lower-is-better). Favors fused
    /// implementations over dispatch-heavy ones.
    Dispatches,
}

impl Metric {
    /// Extract the metric value from a [`TuningRecord`]. Returns `u64::MAX`
    /// when the underlying sample is missing so missing-metric candidates
    /// fall through to the next-best and never win on this axis.
    pub fn extract(self, rec: &TuningRecord) -> u64 {
        match self {
            Metric::P95 => rec.p95_ns,
            Metric::P99 => rec.p99_ns,
            Metric::EnergyPerOp => rec
                .median_energy_j
                .map(|j| (j * 1_000_000.0) as u64)
                .unwrap_or(u64::MAX),
            Metric::Dispatches => rec.median_dispatches.unwrap_or(u32::MAX) as u64,
        }
    }
}
/// Why the selector rejected a candidate. The `candidate` field carries the
/// id so traces and explanations are unambiguous when several candidates
/// share a rejection category.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RejectionRecord {
    pub candidate: CandidateId,
    pub reason: RejectionReason,
}

/// The reason category for a [`RejectionRecord`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    /// No [`crate::quality::QualityEvidence`] attached to the tuning record
    /// and the active policy requires one. The string is a human-readable
    /// explanation referencing the failing gate family.
    MissingQualityEvidence(String),
    /// A [`crate::quality::QualityEvidence`] was attached but failed a gate.
    /// The string identifies which gate failed and the observed vs.
    /// required score.
    QualityGateFailed {
        gate: String,
        observed: f64,
        threshold: f64,
    },
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
            RejectionReason::MissingQualityEvidence(why) => {
                format!("missing quality evidence: {}", why)
            }
            RejectionReason::QualityGateFailed { gate, observed, threshold } => format!(
                "quality gate '{}' failed (observed={:.4}, threshold={:.4})",
                gate, observed, threshold
            ),
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
        tuning: Box<TuningRecord>,
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