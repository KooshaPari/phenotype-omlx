//! VCS port — version control system abstraction.
//!
//! Traceability: FR-010, FR-014, FR-017 / WP05-T026

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::DomainError;

/// Metadata about an active git worktree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorktreeInfo {
    pub path: PathBuf,
    pub commit: String,
    pub branch: String,
    pub feature_slug: String,
    pub wp_id: String,
}

/// Metadata about a git branch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchInfo {
    pub name: String,
    pub commit: String,
    pub is_remote: bool,
}

/// Result of a merge operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    pub success: bool,
    pub conflicts: Vec<ConflictInfo>,
    pub merged_commit: Option<String>,
    pub commit: Option<String>,
    pub message: Option<String>,
}

/// Description of a merge conflict in a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictInfo {
    pub path: String,
    pub file_path: String,
    pub conflict_type: String,
    pub ours: Option<String>,
    pub theirs: Option<String>,
}

/// Collected feature artifacts discovered in the repository.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureArtifacts {
    pub spec: Option<String>,
    pub research: Option<String>,
    pub plan: Option<String>,
    pub other: Vec<String>,
    pub meta_json: Option<String>,
    pub audit_chain: Option<String>,
    pub evidence_paths: Vec<String>,
}

/// Port for version control system operations.
///
/// Abstracts git so tests can use an in-memory mock.
/// The Git adapter (WP07) implements this with `git2`.
#[async_trait]
pub trait VcsPort: Send + Sync {
    // -- Worktree operations (FR-010) --

    /// Create a worktree for a feature work package, returning its absolute path.
    async fn create_worktree(
        &self,
        feature_slug: &str,
        wp_id: &str,
    ) -> Result<PathBuf, DomainError>;

    async fn list_worktrees(&self) -> Result<Vec<WorktreeInfo>, DomainError>;

    async fn cleanup_worktree(&self, worktree_path: &Path) -> Result<(), DomainError>;

    async fn create_branch(&self, branch_name: &str, base: &str) -> Result<(), DomainError>;

    async fn list_branches(
        &self,
        pattern: Option<&str>,
        remote: bool,
    ) -> Result<Vec<BranchInfo>, DomainError>;

    async fn delete_branch(
        &self,
        branch_name: &str,
        force: bool,
        remote: Option<&str>,
    ) -> Result<(), DomainError>;

    async fn checkout_branch(&self, branch_name: &str) -> Result<(), DomainError>;

    async fn merge_to_target(&self, source: &str, target: &str)
    -> Result<MergeResult, DomainError>;

    async fn detect_conflicts(
        &self,
        source: &str,
        target: &str,
    ) -> Result<Vec<ConflictInfo>, DomainError>;

    async fn read_artifact(
        &self,
        feature_slug: &str,
        relative_path: &str,
    ) -> Result<String, DomainError>;

    async fn write_artifact(
        &self,
        feature_slug: &str,
        relative_path: &str,
        content: &str,
    ) -> Result<(), DomainError>;

    async fn artifact_exists(
        &self,
        feature_slug: &str,
        relative_path: &str,
    ) -> Result<bool, DomainError>;

    async fn scan_feature_artifacts(
        &self,
        feature_slug: &str,
    ) -> Result<FeatureArtifacts, DomainError>;
}
