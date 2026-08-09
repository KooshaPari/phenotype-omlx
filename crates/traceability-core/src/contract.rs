//! `AcceptanceContract` and `ProgressionGate` — **new** hybrid types for this crate.
//!
//! See `docs/adr/ADR-0001-superset-merge.md` §6 for design rationale.
//!
//! * [`AcceptanceContract`] is satisfied **only** when every [`Criterion`] maps to
//!   a [`CoverageState::Covered`] cell in the supplied [`CoverageMatrix`].
//! * [`ProgressionGate`] evaluates layer-to-layer advancement using governance
//!   vocabulary (`not_approved`, `missing_acceptance`, `missing_evidence`,
//!   `missing_implementation`, `missing_test`).

use serde::{Deserialize, Serialize};

use crate::artifact::ArtifactRef;
use crate::governance::Evidence;
use crate::matrix::{CoverageMatrix, CoverageState};
use crate::requirement::{Requirement, RequirementStatus, VerificationMethod};

/// Phenotype layer stack for progression gates.
///
/// Intent → IntentDoc → SpecAdr → PlanWbs → Execution → Evidence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layer {
    Intent,
    IntentDoc,
    SpecAdr,
    PlanWbs,
    Execution,
    Evidence,
}

impl Layer {
    /// Next layer in the canonical stack, if any.
    pub fn next(self) -> Option<Self> {
        match self {
            Self::Intent => Some(Self::IntentDoc),
            Self::IntentDoc => Some(Self::SpecAdr),
            Self::SpecAdr => Some(Self::PlanWbs),
            Self::PlanWbs => Some(Self::Execution),
            Self::Execution => Some(Self::Evidence),
            Self::Evidence => None,
        }
    }

    /// All layers in order.
    pub fn all() -> &'static [Self] {
        &[
            Self::Intent,
            Self::IntentDoc,
            Self::SpecAdr,
            Self::PlanWbs,
            Self::Execution,
            Self::Evidence,
        ]
    }
}

/// A testable acceptance criterion that must map to a Covered matrix cell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Criterion {
    /// Stable criterion id (e.g. `AC-1`).
    pub id: String,
    /// Test artifact reference (matrix `from` key).
    pub test_ref: String,
    /// Evidence / requirement reference (matrix `to` key).
    pub evidence_ref: String,
}

/// Gherkin scenario reference for BDD traceability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GherkinRef {
    pub feature_file: String,
    pub scenario: String,
    pub line: Option<u32>,
}

/// Acceptance contract bound to an artifact.
///
/// Satisfied only when **every** criterion resolves to `CoverageState::Covered`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcceptanceContract {
    pub artifact_ref: ArtifactRef,
    pub criteria: Vec<Criterion>,
    pub verification: VerificationMethod,
    pub bdd: Vec<GherkinRef>,
}

impl AcceptanceContract {
    /// Returns `true` when all criteria map to Covered cells in `matrix`.
    ///
    /// Cells are looked up by `(test_ref, evidence_ref)` string keys matching
    /// `MatrixCell::from` / `MatrixCell::to`.
    pub fn is_satisfied(&self, matrix: &CoverageMatrix) -> bool {
        if self.criteria.is_empty() {
            return false;
        }
        self.criteria.iter().all(|c| {
            matrix
                .cells
                .get(&(c.test_ref.clone(), c.evidence_ref.clone()))
                .is_some_and(|cell| cell.coverage == CoverageState::Covered)
        })
    }

    /// Returns criterion ids that are not Covered (or missing from the matrix).
    pub fn unsatisfied_criteria(&self, matrix: &CoverageMatrix) -> Vec<String> {
        self.criteria
            .iter()
            .filter(|c| {
                !matrix
                    .cells
                    .get(&(c.test_ref.clone(), c.evidence_ref.clone()))
                    .is_some_and(|cell| cell.coverage == CoverageState::Covered)
            })
            .map(|c| c.id.clone())
            .collect()
    }
}

/// Predicate vocabulary for progression gates (AgilePlus governance alignment).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GatePredicate {
    NotApproved,
    MissingAcceptance,
    MissingEvidence,
    MissingImplementation,
    MissingTest,
}

/// Reason a gate blocked progression.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateReason {
    NotApproved,
    MissingAcceptance,
    MissingEvidence,
    MissingImplementation,
    MissingTest,
}

impl From<GatePredicate> for GateReason {
    fn from(p: GatePredicate) -> Self {
        match p {
            GatePredicate::NotApproved => Self::NotApproved,
            GatePredicate::MissingAcceptance => Self::MissingAcceptance,
            GatePredicate::MissingEvidence => Self::MissingEvidence,
            GatePredicate::MissingImplementation => Self::MissingImplementation,
            GatePredicate::MissingTest => Self::MissingTest,
        }
    }
}

/// Context supplied when evaluating a [`ProgressionGate`].
#[derive(Debug, Clone, Default)]
pub struct GateContext<'a> {
    /// Requirement under evaluation (if applicable).
    pub requirement: Option<&'a Requirement>,
    /// Acceptance contract for the artifact (if any).
    pub acceptance: Option<&'a AcceptanceContract>,
    /// Coverage matrix for acceptance satisfaction.
    pub matrix: Option<&'a CoverageMatrix>,
    /// Evidence artifacts collected for the work package.
    pub evidence: &'a [Evidence],
    /// Whether implementation links exist (Implements trace links present).
    pub has_implementation: bool,
    /// Whether test / Verifies links exist.
    pub has_test_links: bool,
}

/// Layer-to-layer gate with governance predicates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgressionGate {
    pub from_layer: Layer,
    pub to_layer: Layer,
    pub predicates: Vec<GatePredicate>,
}

impl ProgressionGate {
    /// Evaluate whether progression from `from_layer` to `to_layer` is allowed.
    ///
    /// Returns `Ok(())` when all predicates pass; otherwise the first failing
    /// [`GateReason`].
    pub fn evaluate(&self, ctx: &GateContext<'_>) -> Result<(), GateReason> {
        for predicate in &self.predicates {
            if let Some(reason) = Self::check_predicate(*predicate, ctx) {
                return Err(reason);
            }
        }
        Ok(())
    }

    fn check_predicate(predicate: GatePredicate, ctx: &GateContext<'_>) -> Option<GateReason> {
        match predicate {
            GatePredicate::NotApproved => {
                let approved = ctx
                    .requirement
                    .map(|r| {
                        matches!(
                            r.status,
                            RequirementStatus::Approved
                                | RequirementStatus::Implemented
                                | RequirementStatus::Verified
                        )
                    })
                    .unwrap_or(false);
                if !approved {
                    Some(GateReason::NotApproved)
                } else {
                    None
                }
            }
            GatePredicate::MissingAcceptance => {
                let satisfied = ctx
                    .acceptance
                    .zip(ctx.matrix)
                    .map(|(contract, matrix)| contract.is_satisfied(matrix))
                    .unwrap_or(false);
                if !satisfied {
                    Some(GateReason::MissingAcceptance)
                } else {
                    None
                }
            }
            GatePredicate::MissingEvidence => {
                let has_evidence = !ctx.evidence.is_empty();
                if !has_evidence {
                    Some(GateReason::MissingEvidence)
                } else {
                    None
                }
            }
            GatePredicate::MissingImplementation => {
                if !ctx.has_implementation {
                    Some(GateReason::MissingImplementation)
                } else {
                    None
                }
            }
            GatePredicate::MissingTest => {
                if !ctx.has_test_links {
                    Some(GateReason::MissingTest)
                } else {
                    None
                }
            }
        }
    }

    /// Standard gate from Intent → IntentDoc (approval required).
    pub fn intent_to_intent_doc() -> Self {
        Self {
            from_layer: Layer::Intent,
            to_layer: Layer::IntentDoc,
            predicates: vec![GatePredicate::NotApproved],
        }
    }

    /// Standard gate from Execution → Evidence (acceptance + evidence + tests).
    pub fn execution_to_evidence() -> Self {
        Self {
            from_layer: Layer::Execution,
            to_layer: Layer::Evidence,
            predicates: vec![
                GatePredicate::MissingAcceptance,
                GatePredicate::MissingEvidence,
                GatePredicate::MissingTest,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::Artifact;
    use crate::artifact::ArtifactKind;
    use crate::governance::EvidenceType;
    use crate::ids::RequirementId;
    use crate::matrix::MatrixCell;
    use chrono::Utc;
    use indexmap::IndexMap;
    use std::collections::BTreeMap;
    use uuid::Uuid;

    fn sample_matrix(covered: bool) -> CoverageMatrix {
        let state = if covered {
            CoverageState::Covered
        } else {
            CoverageState::Partial
        };
        let mut cells = IndexMap::new();
        cells.insert(
            ("test:T-1".to_string(), "FR-1".to_string()),
            MatrixCell {
                from: "test:T-1".to_string(),
                to: "FR-1".to_string(),
                trace_links: vec![],
                coverage: state,
            },
        );
        CoverageMatrix {
            cells,
            generated_at: Utc::now(),
        }
    }

    fn sample_contract() -> AcceptanceContract {
        AcceptanceContract {
            artifact_ref: ArtifactRef::Requirement {
                id: RequirementId::from_string("FR-1"),
            },
            criteria: vec![Criterion {
                id: "AC-1".to_string(),
                test_ref: "test:T-1".to_string(),
                evidence_ref: "FR-1".to_string(),
            }],
            verification: VerificationMethod::Test,
            bdd: vec![GherkinRef {
                feature_file: "features/auth.feature".to_string(),
                scenario: "User logs in".to_string(),
                line: Some(12),
            }],
        }
    }

    fn approved_requirement() -> Requirement {
        let artifact = Artifact {
            id: Uuid::new_v4(),
            project_id: Uuid::new_v4(),
            kind: ArtifactKind::Requirement,
            title: "FR-1".to_string(),
            description: None,
            external_id: Some("FR-1".to_string()),
            metadata: BTreeMap::new(),
            created_at: None,
            updated_at: None,
        };
        let mut req = Requirement::new(artifact).unwrap();
        req.status = RequirementStatus::Approved;
        req
    }

    #[test]
    fn acceptance_contract_satisfied_when_all_covered() {
        let contract = sample_contract();
        let matrix = sample_matrix(true);
        assert!(contract.is_satisfied(&matrix));
        assert!(contract.unsatisfied_criteria(&matrix).is_empty());
    }

    #[test]
    fn acceptance_contract_not_satisfied_when_partial() {
        let contract = sample_contract();
        let matrix = sample_matrix(false);
        assert!(!contract.is_satisfied(&matrix));
        assert_eq!(contract.unsatisfied_criteria(&matrix), vec!["AC-1"]);
    }

    #[test]
    fn acceptance_contract_empty_criteria_not_satisfied() {
        let contract = AcceptanceContract {
            criteria: vec![],
            ..sample_contract()
        };
        assert!(!contract.is_satisfied(&sample_matrix(true)));
    }

    #[test]
    fn progression_gate_blocks_not_approved() {
        let gate = ProgressionGate::intent_to_intent_doc();
        let mut req = approved_requirement();
        req.status = RequirementStatus::Draft;
        let ctx = GateContext {
            requirement: Some(&req),
            ..Default::default()
        };
        assert_eq!(gate.evaluate(&ctx), Err(GateReason::NotApproved));
    }

    #[test]
    fn progression_gate_passes_when_approved() {
        let gate = ProgressionGate::intent_to_intent_doc();
        let req = approved_requirement();
        let ctx = GateContext {
            requirement: Some(&req),
            ..Default::default()
        };
        assert!(gate.evaluate(&ctx).is_ok());
    }

    #[test]
    fn progression_gate_execution_to_evidence_checks() {
        let gate = ProgressionGate::execution_to_evidence();
        let contract = sample_contract();
        let matrix = sample_matrix(true);
        let evidence = Evidence {
            id: 1,
            wp_id: 1,
            fr_id: "FR-1".to_string(),
            evidence_type: EvidenceType::TestResult,
            artifact_path: "/ci/out.xml".to_string(),
            metadata: None,
            created_at: Utc::now(),
        };
        let evidence_slice = [evidence];
        let ctx = GateContext {
            acceptance: Some(&contract),
            matrix: Some(&matrix),
            evidence: &evidence_slice,
            has_test_links: true,
            ..Default::default()
        };
        assert!(gate.evaluate(&ctx).is_ok());

        let ctx_no_test = GateContext {
            acceptance: Some(&contract),
            matrix: Some(&matrix),
            evidence: &evidence_slice,
            has_test_links: false,
            ..Default::default()
        };
        assert_eq!(gate.evaluate(&ctx_no_test), Err(GateReason::MissingTest));
    }

    #[test]
    fn layer_next_chain() {
        let layers = Layer::all();
        assert_eq!(layers.len(), 6);
        assert_eq!(Layer::Intent.next(), Some(Layer::IntentDoc));
        assert_eq!(Layer::Evidence.next(), None);
    }
}
