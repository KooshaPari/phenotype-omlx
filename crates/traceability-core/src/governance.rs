//! Governance types — contracts, rules, and evidence.
//!
//! Source: [`AgilePlus/crates/agileplus-domain/src/domain/governance.rs`](https://example.invalid/AgilePlus/crates/agileplus-domain/src/domain/governance.rs)
//! (1:1 port of vocabulary; `fr_id` stays `String` at the boundary per ADR-0001 §2).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Policy domain category.
///
/// Source: AgilePlus `governance.rs:7-15`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PolicyDomain {
    Security,
    Quality,
    Compliance,
    Performance,
    Custom,
}

impl PolicyDomain {
    /// Stable lowercase string for SQL / JSON round-trips.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Security => "security",
            Self::Quality => "quality",
            Self::Compliance => "compliance",
            Self::Performance => "performance",
            Self::Custom => "custom",
        }
    }
}

/// The definition of a policy rule (stored as JSON blob).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDefinition {
    pub description: String,
    pub check: PolicyCheck,
}

/// An active policy rule in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    pub id: i64,
    pub domain: PolicyDomain,
    pub rule: PolicyDefinition,
    pub active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A required-evidence entry inside a governance rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRequirement {
    /// Functional-requirement ID the evidence must satisfy.
    pub fr_id: String,
    /// Type of evidence required.
    pub evidence_type: EvidenceType,
}

/// A governance rule captured inside a contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceRule {
    pub transition: String,
    pub required_evidence: Vec<EvidenceRequirement>,
    pub policy_refs: Vec<i64>,
}

/// A versioned governance contract bound to a feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceContract {
    pub id: i64,
    pub feature_id: i64,
    pub version: i32,
    pub rules: Vec<GovernanceRule>,
    pub bound_at: DateTime<Utc>,
}

/// Type of evidence artifact.
///
/// Source: AgilePlus `governance.rs:74-84`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceType {
    TestResult,
    CiOutput,
    ReviewApproval,
    SecurityScan,
    LintResult,
    ManualAttestation,
}

impl EvidenceType {
    /// Stable snake_case string.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TestResult => "test_result",
            Self::CiOutput => "ci_output",
            Self::ReviewApproval => "review_approval",
            Self::SecurityScan => "security_scan",
            Self::LintResult => "lint_result",
            Self::ManualAttestation => "manual_attestation",
        }
    }
}

/// An evidence artifact attached to a work package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub id: i64,
    pub wp_id: i64,
    pub fr_id: String,
    pub evidence_type: EvidenceType,
    pub artifact_path: String,
    pub metadata: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
}

/// The result of a policy check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyCheck {
    ManualApproval,
    Automated,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_domain_as_str() {
        assert_eq!(PolicyDomain::Security.as_str(), "security");
        assert_eq!(PolicyDomain::Quality.as_str(), "quality");
        assert_eq!(PolicyDomain::Compliance.as_str(), "compliance");
        assert_eq!(PolicyDomain::Performance.as_str(), "performance");
        assert_eq!(PolicyDomain::Custom.as_str(), "custom");
    }

    #[test]
    fn evidence_type_as_str() {
        assert_eq!(EvidenceType::TestResult.as_str(), "test_result");
        assert_eq!(EvidenceType::CiOutput.as_str(), "ci_output");
        assert_eq!(
            EvidenceType::ManualAttestation.as_str(),
            "manual_attestation"
        );
    }

    #[test]
    fn governance_contract_serde_roundtrip() {
        let now = Utc::now();
        let contract = GovernanceContract {
            id: 42,
            feature_id: 100,
            version: 3,
            rules: vec![GovernanceRule {
                transition: "Active->Done".to_string(),
                required_evidence: vec![EvidenceRequirement {
                    fr_id: "FR-001".to_string(),
                    evidence_type: EvidenceType::TestResult,
                }],
                policy_refs: vec![1],
            }],
            bound_at: now,
        };
        let json = serde_json::to_string(&contract).unwrap();
        let back: GovernanceContract = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, contract.id);
        assert_eq!(back.rules.len(), 1);
    }
}
