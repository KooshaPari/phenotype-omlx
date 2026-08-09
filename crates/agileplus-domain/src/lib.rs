// SPDX-License-Identifier: MIT OR Apache-2.0
//! `agileplus-domain` — core domain types, error, and port traits.

pub use error::DomainError;
pub use error::ErrorCode;
pub type DomainResult<T> = std::result::Result<T, DomainError>;

pub mod adapters;
pub mod builder;
pub mod config;
pub mod credentials;
pub mod domain;
pub mod error;
pub mod ids;
pub mod intent_graph;
pub mod ports;
pub mod traceability;

// Shared PM/traceability spine (phenotype-pm-core). AgilePlus-local aggregates
// remain in `domain::*`; lifecycle, governance, and intent graph are canonical
// in `traceability-core` and re-exported here for backward-compatible paths.
pub use traceability_core::governance::{
    Evidence, EvidenceRequirement, EvidenceType, GovernanceContract, GovernanceRule, PolicyCheck,
    PolicyDefinition, PolicyDomain, PolicyRule,
};
pub use traceability_core::intent_graph::{
    CanonicalLinkType, CanonicalMap, DagStage, Edge, GraphMetadata, IntentGraph, Meta, Node,
    NodeType, RelationshipType, Status as NodeStatus, ValidationError,
};
pub use traceability_core::lifecycle::{FeatureState, Transition, TransitionResult};
