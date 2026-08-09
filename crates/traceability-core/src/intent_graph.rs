//! Intent Graph ontology — type-safe graph model for traceability of user intent
//! through features, tasks, code, tests, PRs, and artifacts.
//!
//! Source: [`AgilePlus/crates/agileplus-domain/src/intent_graph.rs`](https://example.invalid/AgilePlus/crates/agileplus-domain/src/intent_graph.rs)
//! (1:1 port; `builder` re-exports omitted — not part of this crate).

use std::collections::{HashMap, HashSet};
use std::fmt;

use chrono::{DateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// ---------------------------------------------------------------------------
/// Node Type
/// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum NodeType {
    Intent,
    Plan,
    Feature,
    Story,
    Task,
    Spec,
    Commit,
    Test,
    PR,
    Bug,
    Artifact,
}

impl fmt::Display for NodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            NodeType::Intent => "Intent",
            NodeType::Plan => "Plan",
            NodeType::Feature => "Feature",
            NodeType::Story => "Story",
            NodeType::Task => "Task",
            NodeType::Spec => "Spec",
            NodeType::Commit => "Commit",
            NodeType::Test => "Test",
            NodeType::PR => "PR",
            NodeType::Bug => "Bug",
            NodeType::Artifact => "Artifact",
        };
        write!(f, "{s}")
    }
}

impl From<&str> for NodeType {
    fn from(s: &str) -> Self {
        match s {
            "Intent" => NodeType::Intent,
            "Plan" => NodeType::Plan,
            "Feature" => NodeType::Feature,
            "Story" => NodeType::Story,
            "Task" => NodeType::Task,
            "Spec" => NodeType::Spec,
            "Commit" => NodeType::Commit,
            "Test" => NodeType::Test,
            "PR" => NodeType::PR,
            "Bug" => NodeType::Bug,
            "Artifact" => NodeType::Artifact,
            _ => panic!("unknown NodeType: {s}"),
        }
    }
}

impl TryFrom<String> for NodeType {
    type Error = ValidationError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        match s.as_str() {
            "Intent" => Ok(NodeType::Intent),
            "Plan" => Ok(NodeType::Plan),
            "Feature" => Ok(NodeType::Feature),
            "Story" => Ok(NodeType::Story),
            "Task" => Ok(NodeType::Task),
            "Spec" => Ok(NodeType::Spec),
            "Commit" => Ok(NodeType::Commit),
            "Test" => Ok(NodeType::Test),
            "PR" => Ok(NodeType::PR),
            "Bug" => Ok(NodeType::Bug),
            "Artifact" => Ok(NodeType::Artifact),
            _ => Err(ValidationError::UnknownNodeType(s)),
        }
    }
}

/// ---------------------------------------------------------------------------
/// DAG Stage
/// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DagStage {
    Intent,
    Plan,
    Feature,
    Story,
    Task,
    Spec,
    Commit,
    Test,
    PR,
    Bug,
    Artifact,
}

impl fmt::Display for DagStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DagStage::Intent => "intent",
            DagStage::Plan => "plan",
            DagStage::Feature => "feature",
            DagStage::Story => "story",
            DagStage::Task => "task",
            DagStage::Spec => "spec",
            DagStage::Commit => "commit",
            DagStage::Test => "test",
            DagStage::PR => "pr",
            DagStage::Bug => "bug",
            DagStage::Artifact => "artifact",
        };
        write!(f, "{s}")
    }
}

impl From<&str> for DagStage {
    fn from(s: &str) -> Self {
        match s {
            "intent" => DagStage::Intent,
            "plan" => DagStage::Plan,
            "feature" => DagStage::Feature,
            "story" => DagStage::Story,
            "task" => DagStage::Task,
            "spec" => DagStage::Spec,
            "commit" => DagStage::Commit,
            "test" => DagStage::Test,
            "pr" => DagStage::PR,
            "bug" => DagStage::Bug,
            "artifact" => DagStage::Artifact,
            _ => panic!("unknown DagStage: {s}"),
        }
    }
}

impl TryFrom<String> for DagStage {
    type Error = ValidationError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        match s.as_str() {
            "intent" => Ok(DagStage::Intent),
            "plan" => Ok(DagStage::Plan),
            "feature" => Ok(DagStage::Feature),
            "story" => Ok(DagStage::Story),
            "task" => Ok(DagStage::Task),
            "spec" => Ok(DagStage::Spec),
            "commit" => Ok(DagStage::Commit),
            "test" => Ok(DagStage::Test),
            "pr" => Ok(DagStage::PR),
            "bug" => Ok(DagStage::Bug),
            "artifact" => Ok(DagStage::Artifact),
            _ => Err(ValidationError::InvalidDagStage(s)),
        }
    }
}

/// ---------------------------------------------------------------------------
/// Relationship Type
/// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RelationshipType {
    Implements,
    Tests,
    Covers,
    TracesTo,
    DerivesFrom,
    Resolves,
    Blocks,
    DependsOn,
}

impl fmt::Display for RelationshipType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            RelationshipType::Implements => "implements",
            RelationshipType::Tests => "tests",
            RelationshipType::Covers => "covers",
            RelationshipType::TracesTo => "traces-to",
            RelationshipType::DerivesFrom => "derives-from",
            RelationshipType::Resolves => "resolves",
            RelationshipType::Blocks => "blocks",
            RelationshipType::DependsOn => "depends-on",
        };
        write!(f, "{s}")
    }
}

impl From<&str> for RelationshipType {
    fn from(s: &str) -> Self {
        match s {
            "implements" => RelationshipType::Implements,
            "tests" => RelationshipType::Tests,
            "covers" => RelationshipType::Covers,
            "traces-to" => RelationshipType::TracesTo,
            "derives-from" => RelationshipType::DerivesFrom,
            "resolves" => RelationshipType::Resolves,
            "blocks" => RelationshipType::Blocks,
            "depends-on" => RelationshipType::DependsOn,
            _ => panic!("unknown RelationshipType: {s}"),
        }
    }
}

impl TryFrom<String> for RelationshipType {
    type Error = ValidationError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        match s.as_str() {
            "implements" => Ok(RelationshipType::Implements),
            "tests" => Ok(RelationshipType::Tests),
            "covers" => Ok(RelationshipType::Covers),
            "traces-to" => Ok(RelationshipType::TracesTo),
            "derives-from" => Ok(RelationshipType::DerivesFrom),
            "resolves" => Ok(RelationshipType::Resolves),
            "blocks" => Ok(RelationshipType::Blocks),
            "depends-on" => Ok(RelationshipType::DependsOn),
            _ => Err(ValidationError::UnknownRelationshipType(s)),
        }
    }
}

/// ---------------------------------------------------------------------------
/// Canonical Link Type
/// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalLinkType {
    ParentOf,
    ChildOf,
    DependsOn,
    Blocks,
    Implements,
    Verifies,
    References,
    Duplicates,
}

impl fmt::Display for CanonicalLinkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            CanonicalLinkType::ParentOf => "parent_of",
            CanonicalLinkType::ChildOf => "child_of",
            CanonicalLinkType::DependsOn => "depends_on",
            CanonicalLinkType::Blocks => "blocks",
            CanonicalLinkType::Implements => "implements",
            CanonicalLinkType::Verifies => "verifies",
            CanonicalLinkType::References => "references",
            CanonicalLinkType::Duplicates => "duplicates",
        };
        write!(f, "{s}")
    }
}

impl From<&str> for CanonicalLinkType {
    fn from(s: &str) -> Self {
        match s {
            "parent_of" => CanonicalLinkType::ParentOf,
            "child_of" => CanonicalLinkType::ChildOf,
            "depends_on" => CanonicalLinkType::DependsOn,
            "blocks" => CanonicalLinkType::Blocks,
            "implements" => CanonicalLinkType::Implements,
            "verifies" => CanonicalLinkType::Verifies,
            "references" => CanonicalLinkType::References,
            "duplicates" => CanonicalLinkType::Duplicates,
            _ => panic!("unknown CanonicalLinkType: {s}"),
        }
    }
}

impl TryFrom<String> for CanonicalLinkType {
    type Error = ValidationError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        match s.as_str() {
            "parent_of" => Ok(CanonicalLinkType::ParentOf),
            "child_of" => Ok(CanonicalLinkType::ChildOf),
            "depends_on" => Ok(CanonicalLinkType::DependsOn),
            "blocks" => Ok(CanonicalLinkType::Blocks),
            "implements" => Ok(CanonicalLinkType::Implements),
            "verifies" => Ok(CanonicalLinkType::Verifies),
            "references" => Ok(CanonicalLinkType::References),
            "duplicates" => Ok(CanonicalLinkType::Duplicates),
            _ => Err(ValidationError::UnknownCanonicalLinkType(s)),
        }
    }
}

/// ---------------------------------------------------------------------------
/// Status
/// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Draft,
    Active,
    Completed,
    Deprecated,
    Rejected,
    Open,
    InProgress,
    Blocked,
    Deferred,
    Cancelled,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Status::Draft => "draft",
            Status::Active => "active",
            Status::Completed => "completed",
            Status::Deprecated => "deprecated",
            Status::Rejected => "rejected",
            Status::Open => "open",
            Status::InProgress => "in_progress",
            Status::Blocked => "blocked",
            Status::Deferred => "deferred",
            Status::Cancelled => "cancelled",
        };
        write!(f, "{s}")
    }
}

impl From<&str> for Status {
    fn from(s: &str) -> Self {
        match s {
            "draft" => Status::Draft,
            "active" => Status::Active,
            "completed" => Status::Completed,
            "deprecated" => Status::Deprecated,
            "rejected" => Status::Rejected,
            "open" => Status::Open,
            "in_progress" => Status::InProgress,
            "blocked" => Status::Blocked,
            "deferred" => Status::Deferred,
            "cancelled" => Status::Cancelled,
            _ => panic!("unknown Status: {s}"),
        }
    }
}

impl TryFrom<String> for Status {
    type Error = ValidationError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        match s.as_str() {
            "draft" => Ok(Status::Draft),
            "active" => Ok(Status::Active),
            "completed" => Ok(Status::Completed),
            "deprecated" => Ok(Status::Deprecated),
            "rejected" => Ok(Status::Rejected),
            "open" => Ok(Status::Open),
            "in_progress" => Ok(Status::InProgress),
            "blocked" => Ok(Status::Blocked),
            "deferred" => Ok(Status::Deferred),
            "cancelled" => Ok(Status::Cancelled),
            _ => Err(ValidationError::UnknownStatus(s)),
        }
    }
}

/// ---------------------------------------------------------------------------
/// Meta
/// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    /// Confidence score from 0.0 to 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Origin of the data (e.g., "user-prompt", "git-log", "agent-inference").
    pub source: String,
    /// ISO 8601 timestamp when this element was created or last modified.
    pub timestamp: DateTime<Utc>,
    /// Identifier of the agent or system that created/modified this element.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
}

/// ---------------------------------------------------------------------------
/// Canonical Map
/// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalMap {
    pub link_type: CanonicalLinkType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<String>,
}

/// ---------------------------------------------------------------------------
/// Node
/// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Globally unique node ID. Format: `<Type>#<slug>`.
    pub id: String,
    pub node_type: NodeType,
    pub dag_stage: DagStage,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub status: Status,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    pub meta: Meta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub table_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub table_id: Option<String>,
}

/// ---------------------------------------------------------------------------
/// Edge
/// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub relationship_type: RelationshipType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_map: Option<CanonicalMap>,
    pub meta: Meta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<serde_json::Value>,
}

/// ---------------------------------------------------------------------------
/// Graph Metadata
/// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    pub version: String,
    pub schema_uri: String,
    pub created_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dag_valid: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_system: Option<String>,
}

/// ---------------------------------------------------------------------------
/// Intent Graph
/// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub metadata: GraphMetadata,
}

/// ---------------------------------------------------------------------------
/// Validation Error
/// ---------------------------------------------------------------------------
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ValidationError {
    #[error("invalid node ID: {0}")]
    InvalidNodeId(String),
    #[error("missing required field: {0}")]
    MissingRequiredField(String),
    #[error("unknown node type: {0}")]
    UnknownNodeType(String),
    #[error("unknown DAG stage: {0}")]
    InvalidDagStage(String),
    #[error("unknown relationship type: {0}")]
    UnknownRelationshipType(String),
    #[error("unknown canonical link type: {0}")]
    UnknownCanonicalLinkType(String),
    #[error("unknown status: {0}")]
    UnknownStatus(String),
    #[error("invalid edge constraint: {relationship} from {from} to {to}")]
    InvalidEdgeConstraint {
        relationship: String,
        from: String,
        to: String,
    },
    #[error("cycle detected in graph")]
    CycleDetected,
    #[error("invalid root node: expected Intent, got {0}")]
    InvalidRootNode(String),
    #[error("missing meta on {0}")]
    MissingMeta(String),
    #[error("duplicate node ID: {0}")]
    DuplicateNodeId(String),
    #[error("orphaned edge: {edge_id} references missing node {node_id}")]
    OrphanedEdge { edge_id: String, node_id: String },
    #[error("confidence out of range: {0}")]
    ConfidenceOutOfRange(f64),
}

/// ---------------------------------------------------------------------------
/// Node ID validation
/// ---------------------------------------------------------------------------
fn node_id_regex() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Z][a-z]+#[a-z0-9\-]+$").unwrap())
}

fn is_valid_node_id(id: &str) -> bool {
    node_id_regex().is_match(id)
}

/// ---------------------------------------------------------------------------
/// Edge constraint rules — encoded from the ontology schema
/// ---------------------------------------------------------------------------
fn allowed_edge_constraints(rel: RelationshipType) -> &'static [(NodeType, NodeType)] {
    use NodeType::*;
    use RelationshipType::*;
    static IMPLEMENTS: &[(NodeType, NodeType)] = &[
        (Intent, Feature),
        (Intent, Story),
        (Feature, Task),
        (Story, Task),
        (Task, Commit),
        (Spec, Feature),
        (Spec, Task),
    ];
    static TESTS: &[(NodeType, NodeType)] = &[
        (Feature, Test),
        (Task, Test),
        (Commit, Test),
        (PR, Test),
        (Bug, Test),
    ];
    static COVERS: &[(NodeType, NodeType)] = &[
        (Feature, Test),
        (Task, Test),
        (Feature, Artifact),
        (Task, Artifact),
        (Spec, Feature),
    ];
    static TRACES_TO: &[(NodeType, NodeType)] = &[]; // wildcard: all allowed
    static DERIVES_FROM: &[(NodeType, NodeType)] = &[
        (Feature, Intent),
        (Story, Intent),
        (Story, Feature),
        (Task, Feature),
        (Task, Story),
        (Task, Bug),
    ];
    static RESOLVES: &[(NodeType, NodeType)] = &[(Bug, Commit), (Bug, PR), (Bug, Task)];
    static BLOCKS: &[(NodeType, NodeType)] = &[
        (Task, Task),
        (Bug, Task),
        (Bug, PR),
        (Task, PR),
        (PR, Feature),
    ];
    static DEPENDS_ON: &[(NodeType, NodeType)] = &[
        (Task, Task),
        (Feature, Feature),
        (Story, Story),
        (PR, PR),
        (Task, Artifact),
    ];
    match rel {
        Implements => IMPLEMENTS,
        Tests => TESTS,
        Covers => COVERS,
        TracesTo => TRACES_TO,
        DerivesFrom => DERIVES_FROM,
        Resolves => RESOLVES,
        Blocks => BLOCKS,
        DependsOn => DEPENDS_ON,
    }
}

/// ---------------------------------------------------------------------------
/// IntentGraph validation
/// ---------------------------------------------------------------------------
impl IntentGraph {
    /// Validate the entire graph, collecting all errors.
    pub fn validate(&self) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        // --- node-level checks ---
        let mut seen_ids = HashSet::new();
        for node in &self.nodes {
            if !is_valid_node_id(&node.id) {
                errors.push(ValidationError::InvalidNodeId(node.id.clone()));
            }
            if !seen_ids.insert(node.id.clone()) {
                errors.push(ValidationError::DuplicateNodeId(node.id.clone()));
            }
            if node.meta.source.trim().is_empty() {
                errors.push(ValidationError::MissingMeta(format!(
                    "node {}: source is empty",
                    node.id
                )));
            }
            if let Some(c) = node.meta.confidence
                && !(0.0..=1.0).contains(&c)
            {
                errors.push(ValidationError::ConfidenceOutOfRange(c));
            }
        }

        // --- edge-level checks ---
        let node_map: HashMap<String, &Node> =
            self.nodes.iter().map(|n| (n.id.clone(), n)).collect();
        for edge in &self.edges {
            if let Some(e) = Self::check_edge_against_node_map(edge, &node_map) {
                errors.push(e);
            }
        }

        // --- DAG checks ---
        if let Err(e) = self.check_dag() {
            errors.push(e);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Check that the graph is a DAG with valid root nodes.
    pub fn check_dag(&self) -> Result<(), ValidationError> {
        let node_map: HashMap<String, &Node> =
            self.nodes.iter().map(|n| (n.id.clone(), n)).collect();

        // Build adjacency list (directed edges: source -> target)
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for edge in &self.edges {
            adj.entry(edge.source.clone())
                .or_default()
                .push(edge.target.clone());
        }

        // Detect cycles using DFS with three-color marking
        #[derive(Clone, Copy)]
        enum Color {
            White,
            Gray,
            Black,
        }
        let mut colors: HashMap<String, Color> = self
            .nodes
            .iter()
            .map(|n| (n.id.clone(), Color::White))
            .collect();

        fn dfs(
            node_id: &str,
            adj: &HashMap<String, Vec<String>>,
            colors: &mut HashMap<String, Color>,
        ) -> Result<(), ValidationError> {
            colors.insert(node_id.to_string(), Color::Gray);
            if let Some(neighbors) = adj.get(node_id) {
                for neighbor in neighbors {
                    match colors.get(neighbor).copied().unwrap_or(Color::White) {
                        Color::White => dfs(neighbor, adj, colors)?,
                        Color::Gray => return Err(ValidationError::CycleDetected),
                        Color::Black => {}
                    }
                }
            }
            colors.insert(node_id.to_string(), Color::Black);
            Ok(())
        }

        for node_id in node_map.keys() {
            if matches!(colors.get(node_id), Some(Color::White)) {
                dfs(node_id, &adj, &mut colors)?;
            }
        }

        // Root nodes must be Intent
        let mut in_degree: HashMap<String, usize> =
            self.nodes.iter().map(|n| (n.id.clone(), 0)).collect();
        for edge in &self.edges {
            *in_degree.entry(edge.target.clone()).or_default() += 1;
        }
        for (node_id, degree) in in_degree {
            if degree == 0 {
                let node = node_map.get(&node_id).ok_or_else(|| {
                    ValidationError::MissingRequiredField(format!(
                        "node {node_id} not found but has in-degree 0"
                    ))
                })?;
                if node.node_type != NodeType::Intent {
                    return Err(ValidationError::InvalidRootNode(node.node_type.to_string()));
                }
            }
        }

        Ok(())
    }

    /// Check edge constraints against the ontology rules.
    pub fn check_edge_constraints(&self) -> Result<(), ValidationError> {
        let node_map: HashMap<String, &Node> =
            self.nodes.iter().map(|n| (n.id.clone(), n)).collect();
        for edge in &self.edges {
            if let Some(e) = Self::check_edge_against_node_map(edge, &node_map) {
                return Err(e);
            }
        }
        Ok(())
    }

    fn check_edge_against_node_map(
        edge: &Edge,
        node_map: &HashMap<String, &Node>,
    ) -> Option<ValidationError> {
        let source_node = node_map.get(&edge.source)?;
        let target_node = node_map.get(&edge.target)?;

        // orphaned edge check
        if source_node.id != edge.source || target_node.id != edge.target {
            // This path is unreachable because we used the map, but keep for safety.
            return Some(ValidationError::OrphanedEdge {
                edge_id: edge.id.clone(),
                node_id: edge.source.clone(),
            });
        }

        // meta source must be non-empty
        if edge.meta.source.trim().is_empty() {
            return Some(ValidationError::MissingMeta(format!(
                "edge {}: source is empty",
                edge.id
            )));
        }

        // confidence range
        if let Some(c) = edge.meta.confidence
            && !(0.0..=1.0).contains(&c)
        {
            return Some(ValidationError::ConfidenceOutOfRange(c));
        }

        // edge constraints
        let allowed = allowed_edge_constraints(edge.relationship_type);
        if edge.relationship_type != RelationshipType::TracesTo {
            let pair = (source_node.node_type, target_node.node_type);
            if !allowed.contains(&pair) {
                return Some(ValidationError::InvalidEdgeConstraint {
                    relationship: edge.relationship_type.to_string(),
                    from: source_node.node_type.to_string(),
                    to: target_node.node_type.to_string(),
                });
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_meta() -> Meta {
        Meta {
            confidence: Some(0.95),
            source: "test".to_string(),
            timestamp: Utc::now(),
            agent_id: None,
        }
    }

    fn sample_node(id: &str, node_type: NodeType, dag_stage: DagStage) -> Node {
        Node {
            id: id.to_string(),
            node_type,
            dag_stage,
            title: "Test".to_string(),
            description: None,
            status: Status::Draft,
            tags: vec![],
            meta: sample_meta(),
            properties: None,
            table_ref: None,
            table_id: None,
        }
    }

    fn sample_edge(id: &str, source: &str, target: &str, rel: RelationshipType) -> Edge {
        Edge {
            id: id.to_string(),
            source: source.to_string(),
            target: target.to_string(),
            relationship_type: rel,
            canonical_map: None,
            meta: sample_meta(),
            properties: None,
        }
    }

    #[test]
    fn node_type_display_and_from_str() {
        assert_eq!(NodeType::Feature.to_string(), "Feature");
        assert_eq!(NodeType::from("Bug"), NodeType::Bug);
        assert_eq!(
            NodeType::try_from("Artifact".to_string()).unwrap(),
            NodeType::Artifact
        );
        assert!(NodeType::try_from("Unknown".to_string()).is_err());
    }

    #[test]
    fn dag_stage_display_and_from_str() {
        assert_eq!(DagStage::Commit.to_string(), "commit");
        assert_eq!(DagStage::from("pr"), DagStage::PR);
        assert_eq!(
            DagStage::try_from("bug".to_string()).unwrap(),
            DagStage::Bug
        );
        assert!(DagStage::try_from("unknown".to_string()).is_err());
    }

    #[test]
    fn relationship_type_display_and_from_str() {
        assert_eq!(RelationshipType::TracesTo.to_string(), "traces-to");
        assert_eq!(
            RelationshipType::from("depends-on"),
            RelationshipType::DependsOn
        );
        assert_eq!(
            RelationshipType::try_from("blocks".to_string()).unwrap(),
            RelationshipType::Blocks
        );
        assert!(RelationshipType::try_from("foo".to_string()).is_err());
    }

    #[test]
    fn canonical_link_type_display_and_from_str() {
        assert_eq!(CanonicalLinkType::ParentOf.to_string(), "parent_of");
        assert_eq!(
            CanonicalLinkType::from("verifies"),
            CanonicalLinkType::Verifies
        );
        assert_eq!(
            CanonicalLinkType::try_from("references".to_string()).unwrap(),
            CanonicalLinkType::References
        );
        assert!(CanonicalLinkType::try_from("foo".to_string()).is_err());
    }

    #[test]
    fn status_display_and_from_str() {
        assert_eq!(Status::InProgress.to_string(), "in_progress");
        assert_eq!(Status::from("blocked"), Status::Blocked);
        assert_eq!(
            Status::try_from("cancelled".to_string()).unwrap(),
            Status::Cancelled
        );
        assert!(Status::try_from("foo".to_string()).is_err());
    }

    #[test]
    fn valid_node_id_passes() {
        assert!(is_valid_node_id("Feature#auth-oauth2"));
        assert!(is_valid_node_id("Bug#123"));
        assert!(is_valid_node_id("Task#fix-memory-leak"));
    }

    #[test]
    fn invalid_node_id_fails() {
        assert!(!is_valid_node_id("feature#auth-oauth2")); // lowercase start
        assert!(!is_valid_node_id("Feature#")); // empty slug
        assert!(!is_valid_node_id("Feature_auth")); // missing #
        assert!(!is_valid_node_id("F#auth")); // single upper letter
    }

    #[test]
    fn validate_ok_for_minimal_graph() {
        let graph = IntentGraph {
            nodes: vec![
                sample_node("Intent#user-auth", NodeType::Intent, DagStage::Intent),
                sample_node("Feature#oauth2", NodeType::Feature, DagStage::Feature),
            ],
            edges: vec![sample_edge(
                "e1",
                "Intent#user-auth",
                "Feature#oauth2",
                RelationshipType::Implements,
            )],
            metadata: GraphMetadata {
                version: "1.0.0".to_string(),
                schema_uri: "https://phenotype.dev/schemas/agileplus-intent-ontology/v1.json"
                    .to_string(),
                created_at: Utc::now(),
                updated_at: None,
                node_count: None,
                edge_count: None,
                dag_valid: None,
                source_system: None,
            },
        };
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn validate_rejects_duplicate_node_id() {
        let graph = IntentGraph {
            nodes: vec![
                sample_node("Intent#user-auth", NodeType::Intent, DagStage::Intent),
                sample_node("Intent#user-auth", NodeType::Intent, DagStage::Intent),
            ],
            edges: vec![],
            metadata: GraphMetadata {
                version: "1.0.0".to_string(),
                schema_uri: "https://phenotype.dev/schemas/agileplus-intent-ontology/v1.json"
                    .to_string(),
                created_at: Utc::now(),
                updated_at: None,
                node_count: None,
                edge_count: None,
                dag_valid: None,
                source_system: None,
            },
        };
        let err = graph.validate().unwrap_err();
        assert!(
            err.iter()
                .any(|e| matches!(e, ValidationError::DuplicateNodeId(_)))
        );
    }

    #[test]
    fn validate_rejects_invalid_node_id() {
        let graph = IntentGraph {
            nodes: vec![sample_node(
                "intent#bad",
                NodeType::Intent,
                DagStage::Intent,
            )],
            edges: vec![],
            metadata: GraphMetadata {
                version: "1.0.0".to_string(),
                schema_uri: "https://phenotype.dev/schemas/agileplus-intent-ontology/v1.json"
                    .to_string(),
                created_at: Utc::now(),
                updated_at: None,
                node_count: None,
                edge_count: None,
                dag_valid: None,
                source_system: None,
            },
        };
        let err = graph.validate().unwrap_err();
        assert!(
            err.iter()
                .any(|e| matches!(e, ValidationError::InvalidNodeId(_)))
        );
    }

    #[test]
    fn dag_detects_cycle() {
        let graph = IntentGraph {
            nodes: vec![
                sample_node("Intent#root", NodeType::Intent, DagStage::Intent),
                sample_node("Feature#a", NodeType::Feature, DagStage::Feature),
                sample_node("Task#b", NodeType::Task, DagStage::Task),
            ],
            edges: vec![
                sample_edge(
                    "e1",
                    "Intent#root",
                    "Feature#a",
                    RelationshipType::Implements,
                ),
                sample_edge("e2", "Feature#a", "Task#b", RelationshipType::Implements),
                sample_edge("e3", "Task#b", "Feature#a", RelationshipType::DependsOn),
            ],
            metadata: GraphMetadata {
                version: "1.0.0".to_string(),
                schema_uri: "https://phenotype.dev/schemas/agileplus-intent-ontology/v1.json"
                    .to_string(),
                created_at: Utc::now(),
                updated_at: None,
                node_count: None,
                edge_count: None,
                dag_valid: None,
                source_system: None,
            },
        };
        let err = graph.check_dag().unwrap_err();
        assert!(matches!(err, ValidationError::CycleDetected));
    }

    #[test]
    fn dag_rejects_non_intent_root() {
        let graph = IntentGraph {
            nodes: vec![
                sample_node("Feature#root", NodeType::Feature, DagStage::Feature),
                sample_node("Task#child", NodeType::Task, DagStage::Task),
            ],
            edges: vec![sample_edge(
                "e1",
                "Feature#root",
                "Task#child",
                RelationshipType::Implements,
            )],
            metadata: GraphMetadata {
                version: "1.0.0".to_string(),
                schema_uri: "https://phenotype.dev/schemas/agileplus-intent-ontology/v1.json"
                    .to_string(),
                created_at: Utc::now(),
                updated_at: None,
                node_count: None,
                edge_count: None,
                dag_valid: None,
                source_system: None,
            },
        };
        let err = graph.check_dag().unwrap_err();
        assert!(matches!(err, ValidationError::InvalidRootNode(_)));
    }

    #[test]
    fn edge_constraint_rejects_invalid_pair() {
        let graph = IntentGraph {
            nodes: vec![
                sample_node("Bug#crash", NodeType::Bug, DagStage::Bug),
                sample_node("Feature#auth", NodeType::Feature, DagStage::Feature),
            ],
            edges: vec![sample_edge(
                "e1",
                "Bug#crash",
                "Feature#auth",
                RelationshipType::Implements,
            )],
            metadata: GraphMetadata {
                version: "1.0.0".to_string(),
                schema_uri: "https://phenotype.dev/schemas/agileplus-intent-ontology/v1.json"
                    .to_string(),
                created_at: Utc::now(),
                updated_at: None,
                node_count: None,
                edge_count: None,
                dag_valid: None,
                source_system: None,
            },
        };
        let err = graph.check_edge_constraints().unwrap_err();
        assert!(matches!(err, ValidationError::InvalidEdgeConstraint { .. }));
    }

    #[test]
    fn traces_to_allows_any_pair() {
        let graph = IntentGraph {
            nodes: vec![
                sample_node("Bug#crash", NodeType::Bug, DagStage::Bug),
                sample_node("Artifact#bin", NodeType::Artifact, DagStage::Artifact),
            ],
            edges: vec![sample_edge(
                "e1",
                "Bug#crash",
                "Artifact#bin",
                RelationshipType::TracesTo,
            )],
            metadata: GraphMetadata {
                version: "1.0.0".to_string(),
                schema_uri: "https://phenotype.dev/schemas/agileplus-intent-ontology/v1.json"
                    .to_string(),
                created_at: Utc::now(),
                updated_at: None,
                node_count: None,
                edge_count: None,
                dag_valid: None,
                source_system: None,
            },
        };
        assert!(graph.check_edge_constraints().is_ok());
    }

    #[test]
    fn validate_rejects_missing_meta_source() {
        let mut graph = IntentGraph {
            nodes: vec![sample_node(
                "Intent#root",
                NodeType::Intent,
                DagStage::Intent,
            )],
            edges: vec![],
            metadata: GraphMetadata {
                version: "1.0.0".to_string(),
                schema_uri: "https://phenotype.dev/schemas/agileplus-intent-ontology/v1.json"
                    .to_string(),
                created_at: Utc::now(),
                updated_at: None,
                node_count: None,
                edge_count: None,
                dag_valid: None,
                source_system: None,
            },
        };
        graph.nodes[0].meta.source = "   ".to_string();
        let err = graph.validate().unwrap_err();
        assert!(
            err.iter()
                .any(|e| matches!(e, ValidationError::MissingMeta(_)))
        );
    }

    #[test]
    fn validate_rejects_confidence_out_of_range() {
        let mut graph = IntentGraph {
            nodes: vec![sample_node(
                "Intent#root",
                NodeType::Intent,
                DagStage::Intent,
            )],
            edges: vec![],
            metadata: GraphMetadata {
                version: "1.0.0".to_string(),
                schema_uri: "https://phenotype.dev/schemas/agileplus-intent-ontology/v1.json"
                    .to_string(),
                created_at: Utc::now(),
                updated_at: None,
                node_count: None,
                edge_count: None,
                dag_valid: None,
                source_system: None,
            },
        };
        graph.nodes[0].meta.confidence = Some(1.5);
        let err = graph.validate().unwrap_err();
        assert!(
            err.iter()
                .any(|e| matches!(e, ValidationError::ConfidenceOutOfRange(_)))
        );
    }
}
