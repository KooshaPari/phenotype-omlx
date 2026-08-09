//! Governance types — re-exported from `traceability-core`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Policy domain category.
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
    pub fn as_str(self) -> &'static str {
        match self {
            PolicyDomain::Security => "security",
            PolicyDomain::Quality => "quality",
            PolicyDomain::Compliance => "compliance",
            PolicyDomain::Performance => "performance",
            PolicyDomain::Custom => "custom",
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

/// A governance rule captured inside a contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceRule {
    pub transition: String,
    pub required_evidence: Vec<String>,
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
    pub fn as_str(self) -> &'static str {
        match self {
            EvidenceType::TestResult => "test_result",
            EvidenceType::CiOutput => "ci_output",
            EvidenceType::ReviewApproval => "review_approval",
            EvidenceType::SecurityScan => "security_scan",
            EvidenceType::LintResult => "lint_result",
            EvidenceType::ManualAttestation => "manual_attestation",
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyCheck {
    ManualApproval,
    Automated,
    EvidencePresent { evidence_type: EvidenceType },
    ThresholdMet { metric: String, min: f64 },
    Custom { script: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use traceability_core::governance::{
        EvidenceRequirement, EvidenceType as TcEvidenceType, GovernanceRule as TcGovernanceRule,
    };

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
        assert_eq!(EvidenceType::ReviewApproval.as_str(), "review_approval");
        assert_eq!(EvidenceType::SecurityScan.as_str(), "security_scan");
        assert_eq!(EvidenceType::LintResult.as_str(), "lint_result");
        assert_eq!(
            EvidenceType::ManualAttestation.as_str(),
            "manual_attestation"
        );
    }

    #[test]
    fn policy_domain_serde_roundtrip() {
        let json = serde_json::to_string(&PolicyDomain::Security).unwrap();
        let back: PolicyDomain = serde_json::from_str(&json).unwrap();
        assert_eq!(back, PolicyDomain::Security);
    }

    #[test]
    fn evidence_type_serde_snake_case() {
        let json = serde_json::to_string(&EvidenceType::TestResult).unwrap();
        assert_eq!(json, "\"test_result\"");
    }

    #[test]
    fn policy_definition_construction_and_serde() {
        let def = PolicyDefinition {
            description: "All tests must pass".to_string(),
            check: PolicyCheck::Automated,
        };
        assert_eq!(def.description, "All tests must pass");
        assert_eq!(def.check, PolicyCheck::Automated);

        let json = serde_json::to_string(&def).unwrap();
        let back: PolicyDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(back.description, def.description);
        assert_eq!(back.check, def.check);
    }

    #[test]
    fn policy_rule_construction_and_serde() {
        let now = Utc::now();
        let rule = PolicyRule {
            id: 1,
            domain: PolicyDomain::Quality,
            rule: PolicyDefinition {
                description: "Test rule".to_string(),
                check: PolicyCheck::ManualApproval,
            },
            active: true,
            created_at: now,
            updated_at: now,
        };
        assert_eq!(rule.id, 1);
        assert_eq!(rule.domain, PolicyDomain::Quality);
        assert!(rule.active);

        let json = serde_json::to_string(&rule).unwrap();
        let back: PolicyRule = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, rule.id);
        assert_eq!(back.domain, rule.domain);
        assert_eq!(back.active, rule.active);
        assert_eq!(back.rule.description, rule.rule.description);
        assert_eq!(back.rule.check, rule.rule.check);
    }

    #[test]
    fn evidence_requirement_construction_and_serde() {
        let req = EvidenceRequirement {
            fr_id: "FR-001".to_string(),
            evidence_type: TcEvidenceType::TestResult,
        };
        assert_eq!(req.fr_id, "FR-001");
        assert_eq!(req.evidence_type, TcEvidenceType::TestResult);

        let json = serde_json::to_string(&req).unwrap();
        let back: EvidenceRequirement = serde_json::from_str(&json).unwrap();
        assert_eq!(back.fr_id, req.fr_id);
        assert_eq!(back.evidence_type, req.evidence_type);
    }

    #[test]
    fn governance_rule_construction_and_serde() {
        let rule = TcGovernanceRule {
            transition: "Draft->Active".to_string(),
            required_evidence: vec![
                EvidenceRequirement {
                    fr_id: "FR-001".to_string(),
                    evidence_type: TcEvidenceType::TestResult,
                },
                EvidenceRequirement {
                    fr_id: "FR-002".to_string(),
                    evidence_type: TcEvidenceType::ReviewApproval,
                },
            ],
            policy_refs: vec![1, 2, 3],
        };
        assert_eq!(rule.transition, "Draft->Active");
        assert_eq!(rule.required_evidence.len(), 2);
        assert_eq!(rule.policy_refs, vec![1, 2, 3]);

        let json = serde_json::to_string(&rule).unwrap();
        let back: TcGovernanceRule = serde_json::from_str(&json).unwrap();
        assert_eq!(back.transition, rule.transition);
        assert_eq!(back.required_evidence.len(), rule.required_evidence.len());
        assert_eq!(back.policy_refs, rule.policy_refs);
    }

    #[test]
    fn governance_contract_construction_and_serde() {
        let now = Utc::now();
        let contract = GovernanceContract {
            id: 42,
            feature_id: 100,
            version: 3,
            rules: vec![GovernanceRule {
                transition: "Active->Done".to_string(),
                required_evidence: vec![],
                policy_refs: vec![1],
            }],
            bound_at: now,
        };
        assert_eq!(contract.id, 42);
        assert_eq!(contract.feature_id, 100);
        assert_eq!(contract.version, 3);
        assert_eq!(contract.rules.len(), 1);

        let json = serde_json::to_string(&contract).unwrap();
        let back: GovernanceContract = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, contract.id);
        assert_eq!(back.feature_id, contract.feature_id);
        assert_eq!(back.version, contract.version);
        assert_eq!(back.rules.len(), contract.rules.len());
    }

    #[test]
    fn evidence_construction_and_serde() {
        let now = Utc::now();
        let evidence = Evidence {
            id: 1,
            wp_id: 5,
            fr_id: "FR-001".to_string(),
            evidence_type: EvidenceType::SecurityScan,
            artifact_path: "/path/to/scan.json".to_string(),
            metadata: Some(serde_json::json!({"scanner": "trivy"})),
            created_at: now,
        };
        assert_eq!(evidence.id, 1);
        assert_eq!(evidence.wp_id, 5);
        assert_eq!(evidence.fr_id, "FR-001");
        assert_eq!(evidence.evidence_type, EvidenceType::SecurityScan);
        assert_eq!(evidence.artifact_path, "/path/to/scan.json");
        assert!(evidence.metadata.is_some());

        let json = serde_json::to_string(&evidence).unwrap();
        let back: Evidence = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, evidence.id);
        assert_eq!(back.wp_id, evidence.wp_id);
        assert_eq!(back.fr_id, evidence.fr_id);
        assert_eq!(back.evidence_type, evidence.evidence_type);
        assert_eq!(back.artifact_path, evidence.artifact_path);
        assert_eq!(back.metadata, evidence.metadata);
    }

    #[test]
    fn evidence_without_metadata_serde() {
        let now = Utc::now();
        let evidence = Evidence {
            id: 2,
            wp_id: 6,
            fr_id: "FR-002".to_string(),
            evidence_type: EvidenceType::LintResult,
            artifact_path: "/path/to/lint.json".to_string(),
            metadata: None,
            created_at: now,
        };
        let json = serde_json::to_string(&evidence).unwrap();
        let back: Evidence = serde_json::from_str(&json).unwrap();
        assert_eq!(back.metadata, None);
    }

    #[test]
    fn policy_check_serde() {
        let manual = PolicyCheck::ManualApproval;
        let automated = PolicyCheck::Automated;

        let json_manual = serde_json::to_string(&manual).unwrap();
        let json_auto = serde_json::to_string(&automated).unwrap();

        let back_manual: PolicyCheck = serde_json::from_str(&json_manual).unwrap();
        let back_auto: PolicyCheck = serde_json::from_str(&json_auto).unwrap();

        assert_eq!(back_manual, PolicyCheck::ManualApproval);
        assert_eq!(back_auto, PolicyCheck::Automated);
    }
}
