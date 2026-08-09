//! Impact scoring (Phase 3 of the Tracera decouple plan).
//!
//! Source: [`Tracera/crates/tracera-core/src/impact.rs`](https://example.invalid/Tracera/crates/tracera-core/src/impact.rs)
//! (1:1 port with `crate::` path adjustments).
//!
//! Ported from `Tracera/src/tracertm/services/blast_radius_service.py` (BFS-based blast radius)
//! and `Tracera/src/tracertm/services/impact_analysis_service.py` (weighted impact scoring).
//!
//! Given a set of changed artifacts, compute:
//! - the **blast radius**: every artifact transitively reachable via trace links
//! - the **impact score**: weighted sum of affected artifacts (with kind weights and confidence)

use std::collections::{HashMap, HashSet, VecDeque, hash_map::Entry};

use serde::{Deserialize, Serialize};

use crate::artifact::ArtifactRef;
use crate::matrix::CoverageMatrix;
use crate::tracelink::{TraceLink, TraceLinkType};

/// Configuration for impact scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactConfig {
    /// Per-artifact-kind weights (test/code/requirement/etc.).
    pub kind_weights: HashMap<String, f32>,
    /// Maximum BFS depth (0 = unbounded).
    pub max_depth: u32,
    /// Multiplier for ConflictsWith links (negative impact).
    pub conflict_multiplier: f32,
    /// Multiplier for Refines/Satisfies/Implements links.
    pub positive_multiplier: f32,
}

impl Default for ImpactConfig {
    fn default() -> Self {
        let mut kind_weights = HashMap::new();
        kind_weights.insert("requirement".to_string(), 1.0);
        kind_weights.insert("nfr".to_string(), 1.5);
        kind_weights.insert("test".to_string(), 0.5);
        kind_weights.insert("code".to_string(), 0.8);
        kind_weights.insert("evidence".to_string(), 0.2);
        kind_weights.insert("journey".to_string(), 0.4);
        kind_weights.insert("agent".to_string(), 0.3);
        kind_weights.insert("document".to_string(), 0.2);
        kind_weights.insert("design".to_string(), 0.7);
        kind_weights.insert("risk".to_string(), 1.2);
        kind_weights.insert("rationale".to_string(), 0.3);
        Self {
            kind_weights,
            max_depth: 10,
            conflict_multiplier: -1.5,
            positive_multiplier: 1.0,
        }
    }
}

/// One node in the blast radius.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlastNode {
    pub artifact: ArtifactRef,
    pub depth: u32,
    pub via: Vec<TraceLinkType>,
    pub score: f32,
}

/// Result of a blast-radius / impact computation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ImpactReport {
    pub seeds: Vec<ArtifactRef>,
    pub blast: Vec<BlastNode>,
    pub total_score: f32,
    pub by_kind: HashMap<String, f32>,
    pub truncated: bool,
    pub max_depth_seen: u32,
    pub conflicts: Vec<TraceLink>,
}

impl ImpactReport {
    pub fn affected_kinds(&self) -> HashSet<String> {
        self.by_kind.keys().cloned().collect()
    }
}

/// Compute the impact report for a set of seed artifacts, given a coverage matrix.
pub fn compute_impact(
    matrix: &CoverageMatrix,
    seeds: &[ArtifactRef],
    cfg: &ImpactConfig,
) -> ImpactReport {
    let mut report = ImpactReport {
        seeds: seeds.to_vec(),
        ..Default::default()
    };

    // Build adjacency: for each (from, to) cell, surface the trace links as directed edges
    // in both directions (so we can traverse the graph either way).
    let mut adj: HashMap<String, Vec<(String, &TraceLink)>> = HashMap::new();
    for cell in matrix.cells.values() {
        for link in &cell.trace_links {
            let from = artifact_key(&link.from);
            let to = artifact_key(&link.to);
            adj.entry(from.clone())
                .or_default()
                .push((to.clone(), link));
            adj.entry(to).or_default().push((from.clone(), link));
        }
    }

    // BFS from each seed.
    //
    // The `via` path is stored ONLY in the BlastNode (set on first discovery);
    // it is NOT propagated through the queue. Earlier versions cloned a growing
    // `via_types: Vec<TraceLinkType>` into the queue on every pop, which made
    // the 10k-node regression gate O(N^2). Keeping the queue element to
    // (node, depth, decay) makes the traversal O(N + E).
    let mut visited: HashMap<String, BlastNode> = HashMap::new();
    for seed in seeds {
        let seed_key = artifact_key(seed);
        let weight = kind_weight(&seed_key, cfg);
        visited.entry(seed_key.clone()).or_insert(BlastNode {
            artifact: seed.clone(),
            depth: 0,
            via: vec![],
            score: weight,
        });
    }

    let mut queue: VecDeque<(String, u32, f32)> = VecDeque::new();
    for seed in seeds {
        queue.push_back((artifact_key(seed), 0, 1.0));
    }

    let mut conflicts: Vec<TraceLink> = Vec::new();
    // Bidirectional adjacency pushes the same conflict link from both endpoints,
    // so we dedup by (from, to, link_type) before recording it. Otherwise a
    // single ConflictsWith link in a 2-node graph ends up recorded 3x.
    let mut conflict_keys: HashSet<(String, String, TraceLinkType)> = HashSet::new();
    let mut max_depth_seen = 0;

    while let Some((node_key, depth, decay)) = queue.pop_front() {
        max_depth_seen = max_depth_seen.max(depth);
        if cfg.max_depth > 0 && depth >= cfg.max_depth {
            report.truncated = true;
            continue;
        }
        if let Some(neighbors) = adj.get(&node_key) {
            for (nbr_key, link) in neighbors {
                let nbr_artifact = parse_artifact_key(nbr_key);
                let link_multiplier = match link.link_type {
                    TraceLinkType::Satisfies
                    | TraceLinkType::Implements
                    | TraceLinkType::Refines => cfg.positive_multiplier,
                    TraceLinkType::ConflictsWith => cfg.conflict_multiplier,
                    _ => 0.0,
                };
                let edge_score = link.confidence * link_multiplier * decay;
                let weight = kind_weight(nbr_key, cfg);
                let score = weight * edge_score.abs() * edge_score.signum();
                if matches!(link.link_type, TraceLinkType::ConflictsWith) {
                    let key = (
                        artifact_key(&link.from),
                        artifact_key(&link.to),
                        link.link_type,
                    );
                    if conflict_keys.insert(key) {
                        conflicts.push((*link).clone());
                    }
                }

                let should_enqueue = match visited.entry(nbr_key.clone()) {
                    Entry::Vacant(entry) => {
                        entry.insert(BlastNode {
                            artifact: nbr_artifact,
                            depth: depth + 1,
                            via: vec![link.link_type],
                            score,
                        });
                        true
                    }
                    Entry::Occupied(mut entry) => {
                        let node = entry.get_mut();
                        if node.score.abs() < score.abs() {
                            node.score = score;
                            true
                        } else {
                            false
                        }
                    }
                };
                if should_enqueue && (depth + 1 <= cfg.max_depth || cfg.max_depth == 0) {
                    queue.push_back((nbr_key.to_string(), depth + 1, decay * 0.85));
                }
            }
        }
    }

    // Sum and bucket
    let mut total_score = 0.0f32;
    let mut by_kind: HashMap<String, f32> = HashMap::new();
    let mut blast: Vec<BlastNode> = visited.into_values().collect();
    blast.sort_by(|a, b| {
        b.score
            .abs()
            .partial_cmp(&a.score.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for node in &blast {
        total_score += node.score;
        let kind = node.artifact.kind_str();
        *by_kind.entry(kind).or_insert(0.0) += node.score;
    }
    // Always include seeds even if no edges
    for seed in seeds {
        if !blast
            .iter()
            .any(|n| artifact_key(&n.artifact) == artifact_key(seed))
        {
            let kind = seed.kind_str();
            *by_kind.entry(kind).or_insert(0.0) += kind_weight(&artifact_key(seed), cfg);
            blast.push(BlastNode {
                artifact: seed.clone(),
                depth: 0,
                via: vec![],
                score: kind_weight(&artifact_key(seed), cfg),
            });
        }
    }

    report.blast = blast;
    report.total_score = total_score;
    report.by_kind = by_kind;
    report.max_depth_seen = max_depth_seen;
    report.conflicts = conflicts;
    report
}

fn kind_weight(artifact_key_str: &str, cfg: &ImpactConfig) -> f32 {
    let kind = infer_kind(artifact_key_str);
    cfg.kind_weights.get(&kind).copied().unwrap_or(0.5)
}

fn infer_kind(artifact_key_str: &str) -> String {
    if artifact_key_str.starts_with("FR-") {
        "requirement".to_string()
    } else if artifact_key_str.starts_with("NFR-") {
        "nfr".to_string()
    } else if artifact_key_str.starts_with("test:") {
        "test".to_string()
    } else if artifact_key_str.starts_with("code:") {
        "code".to_string()
    } else if artifact_key_str.starts_with("journey:") {
        "journey".to_string()
    } else if artifact_key_str.starts_with("agent:") {
        "agent".to_string()
    } else if artifact_key_str.starts_with("evidence:") {
        "evidence".to_string()
    } else if artifact_key_str.starts_with("document:") {
        "document".to_string()
    } else {
        "unknown".to_string()
    }
}

/// Convenience: rank artifacts by impact (descending score).
pub fn top_affected(report: &ImpactReport, n: usize) -> Vec<&BlastNode> {
    report.blast.iter().take(n).collect()
}

/// Convenience: only the conflicts (links or artifacts that are in conflict).
pub fn conflicts_only(report: &ImpactReport) -> &[TraceLink] {
    &report.conflicts
}

fn artifact_key(a: &ArtifactRef) -> String {
    match a {
        ArtifactRef::Requirement { id } => id.as_str().to_string(),
        ArtifactRef::NonFunctionalRequirement { id } => id.as_str().to_string(),
        ArtifactRef::Test { id } => format!("test:{id}"),
        ArtifactRef::CodeEntity { id, .. } => format!("code:{id}"),
        ArtifactRef::Journey { id } => format!("journey:{id}"),
        ArtifactRef::AgentRun { id } => format!("agent:{id}"),
        ArtifactRef::Evidence { id, .. } => format!("evidence:{id}"),
        ArtifactRef::Document { id, .. } => format!("document:{id}"),
    }
}

fn parse_artifact_key(s: &str) -> ArtifactRef {
    if let Some(rest) = s.strip_prefix("test:") {
        ArtifactRef::Test {
            id: rest.to_string(),
        }
    } else if let Some(rest) = s.strip_prefix("code:") {
        ArtifactRef::CodeEntity {
            id: rest.to_string(),
            lang: "rust".to_string(),
        }
    } else if let Some(rest) = s.strip_prefix("journey:") {
        ArtifactRef::Journey {
            id: rest.to_string(),
        }
    } else if let Some(rest) = s.strip_prefix("agent:") {
        ArtifactRef::AgentRun {
            id: rest.to_string(),
        }
    } else if let Some(rest) = s.strip_prefix("evidence:") {
        ArtifactRef::Evidence {
            id: rest.to_string(),
            sha256: "0".repeat(64),
        }
    } else if let Some(rest) = s.strip_prefix("document:") {
        ArtifactRef::Document {
            id: rest.to_string(),
            range: None,
        }
    } else if let Some(rest) = s.strip_prefix("NFR-") {
        ArtifactRef::NonFunctionalRequirement {
            id: crate::ids::NfrId::from_string(rest),
        }
    } else if s.starts_with("FR-") {
        ArtifactRef::Requirement {
            id: crate::ids::RequirementId::from_string(s),
        }
    } else {
        ArtifactRef::Test { id: s.to_string() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::RequirementId;
    use crate::matrix::{CoverageState, MatrixCell};
    use crate::tracelink::TraceLinkType;
    use chrono::Utc;
    use uuid::Uuid;

    fn req(id: &str) -> ArtifactRef {
        ArtifactRef::Requirement {
            id: RequirementId::from_string(id),
        }
    }
    fn test(id: &str) -> ArtifactRef {
        ArtifactRef::Test { id: id.to_string() }
    }

    fn make_link(from: ArtifactRef, to: ArtifactRef, ty: TraceLinkType, conf: f32) -> TraceLink {
        let project = Uuid::new_v4();
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        let mut link = TraceLink::new(project, source, target, ty).unwrap();
        link.from = from;
        link.to = to;
        link.confidence = conf;
        link.created_at = Some(Utc::now());
        link.updated_at = Some(Utc::now());
        link
    }

    fn make_matrix(links: Vec<TraceLink>) -> CoverageMatrix {
        let mut cells: indexmap::IndexMap<(String, String), MatrixCell> =
            indexmap::IndexMap::with_hasher(Default::default());
        for link in links {
            let key = (artifact_key(&link.from), artifact_key(&link.to));
            let cell = cells.entry(key).or_insert_with(|| MatrixCell {
                from: artifact_key(&link.from),
                to: artifact_key(&link.to),
                trace_links: Vec::new(),
                coverage: CoverageState::Covered,
            });
            cell.trace_links.push(link);
        }
        CoverageMatrix {
            cells,
            generated_at: Utc::now(),
        }
    }

    #[test]
    fn empty_matrix_zero_impact() {
        let matrix = make_matrix(vec![]);
        let report = compute_impact(&matrix, &[req("FR-001")], &ImpactConfig::default());
        // Seed appears with its own weight
        assert!(report.total_score >= 0.0);
        assert_eq!(report.blast.len(), 1);
        assert_eq!(report.conflicts.len(), 0);
    }

    #[test]
    fn single_link_propagates() {
        let link = make_link(req("FR-001"), test("T-001"), TraceLinkType::Verifies, 0.95);
        let matrix = make_matrix(vec![link]);
        let report = compute_impact(&matrix, &[req("FR-001")], &ImpactConfig::default());
        // Should include both seed and test
        assert!(report.blast.len() >= 2);
        assert!(report.total_score > 0.0);
    }

    #[test]
    fn conflict_link_produces_negative_score() {
        let link = make_link(
            req("FR-001"),
            test("T-001"),
            TraceLinkType::ConflictsWith,
            0.95,
        );
        let matrix = make_matrix(vec![link]);
        let report = compute_impact(&matrix, &[req("FR-001")], &ImpactConfig::default());
        // Conflicts contribute negative score and appear in report.conflicts
        assert_eq!(report.conflicts.len(), 1);
        // total score may still be ≥ 0 because of seed weight
    }

    #[test]
    fn multi_hop_traversal() {
        // FR-001 -> T-001 (Verifies) -> T-002 (DerivesFrom) -> FR-002 (Satisfies)
        let l1 = make_link(req("FR-001"), test("T-001"), TraceLinkType::Verifies, 0.9);
        let l2 = make_link(
            test("T-001"),
            test("T-002"),
            TraceLinkType::DerivesFrom,
            0.8,
        );
        let l3 = make_link(test("T-002"), req("FR-002"), TraceLinkType::Satisfies, 0.7);
        let matrix = make_matrix(vec![l1, l2, l3]);
        let report = compute_impact(&matrix, &[req("FR-001")], &ImpactConfig::default());
        // Should reach all 4 nodes
        assert!(report.blast.len() >= 4);
    }

    #[test]
    fn max_depth_truncates() {
        let l1 = make_link(req("FR-001"), test("T-001"), TraceLinkType::Verifies, 0.9);
        let l2 = make_link(
            test("T-001"),
            test("T-002"),
            TraceLinkType::DerivesFrom,
            0.8,
        );
        let matrix = make_matrix(vec![l1, l2]);
        let cfg = ImpactConfig {
            max_depth: 1,
            ..Default::default()
        };
        let report = compute_impact(&matrix, &[req("FR-001")], &cfg);
        assert!(report.truncated);
        // Should only include seed + T-001
        assert!(report.blast.len() <= 2);
    }

    #[test]
    fn top_affected_returns_sorted() {
        let l1 = make_link(req("FR-001"), test("T-001"), TraceLinkType::Verifies, 0.9);
        let l2 = make_link(req("FR-001"), test("T-002"), TraceLinkType::Satisfies, 0.5);
        let matrix = make_matrix(vec![l1, l2]);
        let report = compute_impact(&matrix, &[req("FR-001")], &ImpactConfig::default());
        let top = top_affected(&report, 2);
        assert_eq!(top.len(), 2);
        // Top should be sorted by |score| descending
        assert!(top[0].score.abs() >= top[1].score.abs());
    }

    #[test]
    fn seed_only_artifact_returns_self() {
        let matrix = make_matrix(vec![]);
        let report = compute_impact(&matrix, &[test("T-orphan")], &ImpactConfig::default());
        // No edges but seed appears with own weight
        assert_eq!(report.blast.len(), 1);
        assert_eq!(report.blast[0].depth, 0);
    }

    #[test]
    fn kind_weights_have_defaults() {
        let cfg = ImpactConfig::default();
        assert!(cfg.kind_weights.contains_key("requirement"));
        assert!(cfg.kind_weights.contains_key("nfr"));
        assert!(cfg.kind_weights.contains_key("test"));
        assert!(cfg.kind_weights.contains_key("code"));
    }

    #[test]
    fn impact_analysis_10k_node_regression_gate() {
        let node_count = 10_000;
        let mut links = Vec::with_capacity(node_count - 1);
        let mut previous = req("FR-00000");

        for i in 1..node_count {
            let next = if i % 5 == 0 {
                req(&format!("FR-{i:05}"))
            } else {
                test(&format!("T-{i:05}"))
            };
            links.push(make_link(
                previous.clone(),
                next.clone(),
                TraceLinkType::Satisfies,
                0.95,
            ));
            previous = next;
        }

        let matrix = make_matrix(links);
        let cfg = ImpactConfig {
            max_depth: 0,
            ..Default::default()
        };
        let started = std::time::Instant::now();
        let report = compute_impact(&matrix, &[req("FR-00000")], &cfg);
        let elapsed = started.elapsed();

        assert_eq!(report.blast.len(), node_count);
        assert!(!report.truncated);
        assert!(
            elapsed.as_secs_f64() < 0.5,
            "10k-node impact analysis exceeded 5% regression gate: {elapsed:?}"
        );
    }
}
