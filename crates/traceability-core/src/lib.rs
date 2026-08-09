//! `traceability-core` — the shared PM/traceability spine for **phenotype-pm-core**.
//!
//! This crate is a **superset merge** of two prior domains:
//!
//! * [`tracera-core`](https://example.invalid/Tracera/crates/tracera-core) —
//!   canonical `Artifact` / `Requirement` / `TraceLink` / coverage-matrix model,
//!   7 link types with confidence, ISO 29148 + DO-178C vocabulary.
//! * [`agileplus-domain`](https://example.invalid/AgilePlus/crates/agileplus-domain) —
//!   `FeatureState` 8-stage lifecycle, `IntentGraph` ontology with node types /
//!   dag stages / relationship types, `GovernanceContract` + `PolicyRule` +
//!   `EvidenceType` vocabulary.
//!
//! Hybridisation decisions live in
//! [`docs/adr/ADR-0001-superset-merge.md`](https://example.invalid/docs/adr/ADR-0001-superset-merge.md).
//!
//! ## Module map
//!
//! | Module            | Source of truth         | What it owns                                                       |
//! |-------------------|-------------------------|--------------------------------------------------------------------|
//! | [`ids`]           | Tracera                 | `FR-` / `NFR-` id types, `RequirementId`, `NfrId`                 |
//! | [`artifact`]      | Tracera                 | `Artifact`, `ArtifactKind`, `ArtifactRef`                         |
//! | [`requirement`]   | Tracera ⊕ AgilePlus     | `Requirement` + `RequirementStatus` + `VerificationMethod`        |
//! | [`tracelink`]     | Tracera                 | `TraceLink` + 7 link types + confidence                           |
//! | [`matrix`]        | Tracera                 | `CoverageMatrix`, `MatrixCell`, `CoverageState`, build/query/diff |
//! | [`impact`]        | Tracera                 | `ImpactConfig`, `BlastNode`, `ImpactReport`, `compute_impact`     |
//! | [`intent_graph`]  | AgilePlus               | `NodeType`, `DagStage`, `RelationshipType`, `IntentGraph` validate|
//! | [`lifecycle`]     | AgilePlus               | `FeatureState` 8-stage linear state machine                        |
//! | [`governance`]    | AgilePlus               | `GovernanceContract` / `PolicyRule` / `EvidenceType` |
//! | [`contract`]      | **NEW (this crate)**    | `AcceptanceContract` + `ProgressionGate`                           |
//!
//! Consumers:
//! * **AgilePlus** (authoring) imports `lifecycle`, `governance`, `intent_graph`,
//!   `contract`, `requirement`, `artifact`, `ids`.
//! * **Tracera** (live service) imports `artifact`, `requirement`, `tracelink`,
//!   `matrix`, `impact`, `ids`, and *reads* `contract`/`governance` for
//!   gate evaluation but does not author them.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(missing_docs)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::int_plus_one)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]

pub mod artifact;
pub mod contract;
pub mod governance;
pub mod ids;
pub mod impact;
pub mod intent_graph;
pub mod lifecycle;
pub mod matrix;
pub mod requirement;
pub mod tracelink;

pub use artifact::{Artifact, ArtifactKind, ArtifactRef, LinkKind, TraceLinkError};
pub use contract::{AcceptanceContract, Criterion, GateReason, GherkinRef, Layer, ProgressionGate};
pub use governance::{
    Evidence, EvidenceRequirement, EvidenceType, GovernanceContract, GovernanceRule, PolicyCheck,
    PolicyDefinition, PolicyDomain, PolicyRule,
};
pub use ids::{NfrId, RequirementId};
pub use impact::{
    BlastNode, ImpactConfig, ImpactReport, compute_impact, conflicts_only, top_affected,
};
pub use intent_graph::{
    CanonicalLinkType, DagStage, Edge, GraphMetadata, IntentGraph, Meta, Node, NodeType,
    RelationshipType, Status as NodeStatus, ValidationError,
};
pub use lifecycle::{FeatureState, Transition, TransitionResult};
pub use matrix::{
    BuildResult, MatrixCell, build_from_pairs, build_matrix, classify_cell, neighbors,
};
pub use requirement::{Requirement, RequirementStatus, VerificationMethod, is_core_link_type};
pub use tracelink::{
    CORE_TRACE_LINK_TYPES, NEO4J_NODE_LABELS, NEO4J_RELATIONSHIP_TYPES, Neo4jSchema, TraceLink,
    TraceLinkType,
};

// CoverageState is re-exported from the matrix module so the lib-level
// `pub use` list stays compact.
pub use matrix::{CoverageMatrix, CoverageState};
