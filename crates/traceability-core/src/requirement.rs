//! `Requirement` — a traceable requirement artifact with status and verification method.
//!
//! ## Hybridisation (ADR-0001 §4)
//!
//! * The `Requirement` struct itself is **Tracera's** shape
//!   ([`Tracera/crates/tracera-core/src/lib.rs:157-186`](https://example.invalid/Tracera/crates/tracera-core/src/lib.rs)):
//!   it embeds an `Artifact`, holds `status`, `priority`, `rationale`,
//!   `acceptance_criteria: Vec<String>`, and `verification_method: Option<...>`.
//! * `RequirementStatus` is **Tracera's** ISO 29148 § 5.2.8 vocabulary
//!   (Draft / Proposed / Approved / Implemented / Verified / Deprecated / Rejected).
//! * `VerificationMethod` is **Tracera's** DO-178C / IEEE 1012 vocabulary
//!   (Test / Analysis / Inspection / Demonstration / Review).
//!
//! ### How AgilePlus' `Evidence` / `EvidenceType` is reconciled
//!
//! AgilePlus' `Evidence` (governance.rs:99-109) and `EvidenceType`
//! (governance.rs:74-97) describe *what* a verification artifact is
//! (test result, CI output, review approval, security scan, lint result,
//! manual attestation). Tracera's `VerificationMethod` describes *how* a
//! requirement is satisfied (test/analysis/inspection/demonstration/review).
//! These are orthogonal axes, not duplicates:
//!
//! | VerificationMethod | Compatible EvidenceType(s)              |
//! |--------------------|------------------------------------------|
//! | `Test`             | TestResult, CiOutput, LintResult         |
//! | `Analysis`         | SecurityScan (static), ManualAttestation |
//! | `Inspection`       | ManualAttestation, ReviewApproval        |
//! | `Demonstration`    | ManualAttestation                        |
//! | `Review`           | ReviewApproval, ManualAttestation        |
//!
//! The mapping is intentionally left to the consumer (AgilePlus or Tracera)
//! — `Requirement` exposes `verification_method` and the gate logic in
//! `crate::contract` enforces "every `Criterion` must point to a `CoverageState::Covered`
//! matrix cell for the chosen method".
//!
//! ### How the `acceptance_criteria: Vec<String>` is reconciled with
//! `AcceptanceContract::criteria: Vec<Criterion>`
//!
//! Tracera's `Requirement::acceptance_criteria` is a list of free-form
//! strings (one per acceptance bullet). `AcceptanceContract::criteria` in
//! `crate::contract` is a *testable* list of `Criterion { test_ref, evidence_ref }`
//! that maps to a Covered matrix cell. To stay lossless:
//! 1. `Requirement::acceptance_criteria` keeps the **free-form** shape so
//!    any existing Tracera JSON / SQL payload still round-trips.
//! 2. `AcceptanceContract::criteria` is the **testable** layer that the
//!    progression gate enforces. Free-form bullets can be promoted to
//!    `Criterion` lazily.

use serde::{Deserialize, Serialize};

use crate::artifact::{Artifact, ArtifactKind, TraceLinkError};

/// Lifecycle states for a [`Requirement`] (ISO 29148 § 5.2.8).
///
/// Source: Tracera `lib.rs:114-125`. Vocabulary preserved verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RequirementStatus {
    /// Initial draft, not yet circulated.
    Draft,
    /// Submitted for review.
    Proposed,
    /// Approved by the change-control board.
    Approved,
    /// Code/change has been merged that purports to satisfy the requirement.
    Implemented,
    /// Verification has been completed and recorded.
    Verified,
    /// No longer authoritative; kept for traceability history.
    Deprecated,
    /// Rejected by the change-control board.
    Rejected,
}

impl RequirementStatus {
    /// Is this a terminal state (no further transitions expected)?
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Verified | Self::Deprecated | Self::Rejected)
    }

    /// Is this a "work in progress" state (Draft/Proposed/Approved/Implemented)?
    pub fn is_in_progress(self) -> bool {
        !self.is_terminal()
    }
}

/// DO-178C / IEEE 1012 verification methods used on `Verifies` trace links
/// and recorded on a [`Requirement`].
///
/// Source: Tracera `lib.rs:127-136`. Vocabulary preserved verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VerificationMethod {
    /// Executing the system under test and observing outputs.
    Test,
    /// Analytical proof or model-based reasoning.
    Analysis,
    /// Visual / manual examination of an artifact.
    Inspection,
    /// Operational observation by a stakeholder.
    Demonstration,
    /// Peer / formal review of an artifact.
    Review,
}

impl VerificationMethod {
    /// Lowercase string, stable for SQL/Neo4j/JSON-LD round-trips.
    pub fn as_db_str(self) -> &'static str {
        match self {
            Self::Test => "test",
            Self::Analysis => "analysis",
            Self::Inspection => "inspection",
            Self::Demonstration => "demonstration",
            Self::Review => "review",
        }
    }
}

/// A traceable requirement.
///
/// Source: Tracera `lib.rs:156-186`. Fields preserved verbatim — the merge
/// adds **no** new fields to the type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Requirement {
    /// Embedded artifact node (super-type fields).
    #[serde(flatten)]
    pub artifact: Artifact,
    /// Lifecycle state.
    pub status: RequirementStatus,
    /// Optional MoSCoW-style priority (0..=5; convention 0=critical, 5=nice-to-have).
    pub priority: Option<u8>,
    /// Free-form rationale.
    pub rationale: Option<String>,
    /// Free-form acceptance criteria. Promotable to [`crate::contract::Criterion`]
    /// for testable satisfaction — see module-level docs.
    pub acceptance_criteria: Vec<String>,
    /// Selected verification method. Required for `status = Verified`.
    pub verification_method: Option<VerificationMethod>,
}

impl Requirement {
    /// Construct a `Requirement` with sensible defaults; validates that the
    /// embedded artifact's kind is `Requirement`.
    pub fn new(artifact: Artifact) -> Result<Self, TraceLinkError> {
        if artifact.kind != ArtifactKind::Requirement {
            return Err(TraceLinkError::WrongArtifactKind {
                expected: ArtifactKind::Requirement,
                got: artifact.kind,
            });
        }
        Ok(Self {
            artifact,
            status: RequirementStatus::Draft,
            priority: None,
            rationale: None,
            acceptance_criteria: Vec::new(),
            verification_method: None,
        })
    }

    /// Construct a `Requirement` *without* validating the artifact kind.
    ///
    /// Use this only at JSON-deserialisation / migration boundaries where
    /// the caller has already validated the kind. Prefer [`Self::new`] for
    /// all in-process construction.
    pub fn new_unchecked(artifact: Artifact) -> Self {
        Self {
            artifact,
            status: RequirementStatus::Draft,
            priority: None,
            rationale: None,
            acceptance_criteria: Vec::new(),
            verification_method: None,
        }
    }

    /// Promote a free-form acceptance bullet into a testable criterion
    /// (helper for the boundary between Tracera-shaped free-form data and
    /// the testable `AcceptanceContract::criteria`).
    pub fn push_acceptance_criterion(&mut self, criterion: impl Into<String>) {
        self.acceptance_criteria.push(criterion.into());
    }
}

/// Helper re-exported for the `tracelink` module so that `is_core_link_type`
/// is part of the public API and discoverable from `requirement` (where the
/// type was originally defined in Tracera).
pub use crate::tracelink::is_core_link_type;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::RequirementId;
    use std::collections::BTreeMap;
    use uuid::Uuid;

    fn sample_artifact() -> Artifact {
        Artifact {
            id: Uuid::new_v4(),
            project_id: Uuid::new_v4(),
            kind: ArtifactKind::Requirement,
            title: "FR-1 must X".to_string(),
            description: None,
            external_id: Some(RequirementId::from_string("FR-1").to_fr_id_string()),
            metadata: BTreeMap::new(),
            created_at: None,
            updated_at: None,
        }
    }

    #[test]
    fn requirement_new_defaults() {
        let r = Requirement::new(sample_artifact()).unwrap();
        assert_eq!(r.status, RequirementStatus::Draft);
        assert!(r.acceptance_criteria.is_empty());
        assert!(r.verification_method.is_none());
        assert!(r.priority.is_none());
    }

    #[test]
    fn requirement_rejects_wrong_kind() {
        let mut a = sample_artifact();
        a.kind = ArtifactKind::Code;
        let err = Requirement::new(a).unwrap_err();
        assert!(matches!(err, TraceLinkError::WrongArtifactKind { .. }));
    }

    #[test]
    fn requirement_status_terminal() {
        assert!(RequirementStatus::Verified.is_terminal());
        assert!(RequirementStatus::Deprecated.is_terminal());
        assert!(RequirementStatus::Rejected.is_terminal());
        assert!(!RequirementStatus::Draft.is_terminal());
        assert!(!RequirementStatus::Implemented.is_terminal());
    }

    #[test]
    fn requirement_status_in_progress() {
        assert!(RequirementStatus::Draft.is_in_progress());
        assert!(RequirementStatus::Proposed.is_in_progress());
        assert!(RequirementStatus::Approved.is_in_progress());
        assert!(RequirementStatus::Implemented.is_in_progress());
        assert!(!RequirementStatus::Verified.is_in_progress());
    }

    #[test]
    fn verification_method_db_strings() {
        assert_eq!(VerificationMethod::Test.as_db_str(), "test");
        assert_eq!(VerificationMethod::Analysis.as_db_str(), "analysis");
        assert_eq!(VerificationMethod::Inspection.as_db_str(), "inspection");
        assert_eq!(
            VerificationMethod::Demonstration.as_db_str(),
            "demonstration"
        );
        assert_eq!(VerificationMethod::Review.as_db_str(), "review");
    }

    #[test]
    fn requirement_serde_roundtrip() {
        let mut r = Requirement::new(sample_artifact()).unwrap();
        r.status = RequirementStatus::Approved;
        r.priority = Some(2);
        r.rationale = Some("because".to_string());
        r.push_acceptance_criterion("must return 200 on /healthz");
        r.verification_method = Some(VerificationMethod::Test);

        let json = serde_json::to_string(&r).unwrap();
        let back: Requirement = serde_json::from_str(&json).unwrap();
        assert_eq!(back.status, RequirementStatus::Approved);
        assert_eq!(back.priority, Some(2));
        assert_eq!(back.rationale.as_deref(), Some("because"));
        assert_eq!(back.acceptance_criteria.len(), 1);
        assert_eq!(back.verification_method, Some(VerificationMethod::Test));
        assert_eq!(back.artifact.id, r.artifact.id);
    }

    #[test]
    fn new_unchecked_skips_kind_check() {
        let mut a = sample_artifact();
        a.kind = ArtifactKind::Design; // intentionally wrong
        let r = Requirement::new_unchecked(a);
        // No panic; this is a deliberate escape hatch.
        assert_eq!(r.artifact.kind, ArtifactKind::Design);
    }
}
