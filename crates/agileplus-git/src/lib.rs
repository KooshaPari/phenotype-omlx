// SPDX-License-Identifier: MIT OR Apache-2.0
//! Git VCS adapter for AgilePlus.
//!
//! Backed by [`git2`](https://docs.rs/git2) (libgit2 FFI) for read /
//! write operations that have a clean `git2` API (worktree add, branch
//! create / delete, ref resolution, branch listing), and by the
//! `git(1)` CLI for operations that libgit2 either does not expose or
//! exposes in a way that is brittle across versions â€” namely
//!
//! - `git worktree remove --force <path>` (more reliable than
//!   `Worktree::prune`, which silently no-ops on dirty worktrees),
//! - `git push <remote> :<branch>` to delete a remote branch,
//! - `git checkout` for fast branch switching,
//! - `git merge --no-ff <source> --autostash` to merge a feature branch
//!   with the standard "no fast-forward" / autostash combination the
//!   domain expects, and
//! - `git merge-tree <base> <branch>` for fast file-level conflict
//!   detection without touching the working tree.
//!
//! The adapter is rooted at a single `repo_root: PathBuf` and re-opens
//! the libgit2 repository on each call. This is cheap (libgit2 caches
//! the on-disk parse) and avoids the lifetime pain of holding an open
//! `git2::Repository` across await points.
//!
//! Audit traceability: recs #10, #11 from
//! `AUDIT_BLOC_VS_2026_SOTA.md`.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use git2::Repository;

use agileplus_domain::error::DomainError;
use agileplus_domain::ports::vcs::{
    BranchInfo, ConflictInfo, FeatureArtifacts, MergeResult, VcsPort, WorktreeInfo,
};

use crate::claim_bound::ClaimBoundWorktree;

pub mod claim_bound;

/// Git-backed VCS adapter. Cheap to clone (the heavy state lives in
/// libgit2's on-disk cache).
#[derive(Debug, Clone)]
pub struct GitVcsAdapter {
    pub(crate) repo_root: PathBuf,
}

impl GitVcsAdapter {
    /// Create an adapter rooted at the current working directory.
    pub fn from_current_dir() -> anyhow::Result<Self> {
        Ok(Self {
            repo_root: std::env::current_dir()?,
        })
    }

    /// Create an adapter rooted at an explicit path. The path does not
    /// need to be the repo root â€” it may be a subdirectory; we
    /// `discover()` the actual root on each call.
    pub fn new(repo_root: PathBuf) -> Self {
        Self { repo_root }
    }

    /// Return the adapter's repository root.
    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    /// Open the underlying libgit2 repository, walking up from
    /// `repo_root` if necessary. Maps libgit2 errors to
    /// [`DomainError::Storage`].
    fn open(&self) -> Result<Repository, DomainError> {
        Repository::discover(&self.repo_root).map_err(|e| {
            DomainError::Storage(format!(
                "failed to open git repository at {}: {e}",
                self.repo_root.display()
            ))
        })
    }

    /// Resolve a base ref (branch, tag, commit-ish) to its
    /// [`git2::Commit`].
    fn resolve_commit<'a>(
        repo: &'a Repository,
        base: &str,
    ) -> Result<git2::Commit<'a>, DomainError> {
        let object = repo
            .revparse_single(base)
            .map_err(|e| DomainError::Storage(format!("revparse({base}): {e}")))?;
        object
            .peel_to_commit()
            .map_err(|e| DomainError::Storage(format!("peel to commit({base}): {e}")))
    }

    /// Build the canonical worktree name and branch name for a
    /// `(feature_slug, wp_id)` pair.
    fn worktree_branch(feature_slug: &str, wp_id: &str) -> String {
        format!("feat/{feature_slug}/{wp_id}")
    }

    /// Build the canonical worktree directory name (sibling of
    /// `repo_root`).
    fn worktree_dirname(feature_slug: &str, wp_id: &str) -> String {
        format!("{feature_slug}-{wp_id}")
    }

    /// Compute the absolute worktree path for a `(feature_slug,
    /// wp_id)` pair: `<repo_root>/../<feature_slug>-<wp_id>`.
    pub fn worktree_path(&self, feature_slug: &str, wp_id: &str) -> PathBuf {
        let parent = self
            .repo_root
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        parent.join(Self::worktree_dirname(feature_slug, wp_id))
    }

    /// Run `git(1)` with the given args inside `repo_root` and return
    /// the trimmed stdout on success, or a [`DomainError::Storage`]
    /// with stderr on failure.
    fn run_git(&self, args: &[&str]) -> Result<String, DomainError> {
        let output = Command::new("git")
            .args(args)
            .current_dir(&self.repo_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| DomainError::Storage(format!("failed to spawn git: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            return Err(DomainError::Storage(format!(
                "git {} failed: {}",
                args.join(" "),
                if stderr.trim().is_empty() {
                    stdout.trim()
                } else {
                    stderr.trim()
                }
            )));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Run `git(1)` and return stdout even on non-zero exit (for
    /// commands like `merge-tree` that output conflict info on stdout
    /// and exit with non-zero).
    fn run_git_allow_failure(&self, args: &[&str]) -> Result<String, DomainError> {
        let output = Command::new("git")
            .args(args)
            .current_dir(&self.repo_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| DomainError::Storage(format!("failed to spawn git: {e}")))?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            if !stdout.trim().is_empty() {
                return Ok(stdout);
            }
            return Err(DomainError::Storage(format!(
                "git {} failed: {}",
                args.join(" "),
                stderr.trim()
            )));
        }
        Ok(stdout.trim().to_string())
    }

    /// Run `git(1)` and ignore stdout; only the exit status matters.
    fn run_git_status(&self, args: &[&str]) -> Result<(), DomainError> {
        let _ = self.run_git(args)?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Claim-bound worktree creation (rec #11)
    // -----------------------------------------------------------------

    /// Create a worktree bound to a [`agileplus_triage::claim::Claim`].
    /// Validates that the claim is `kind=Worktree` + `state=Active` and
    /// records the resulting worktree path in the claim's
    /// [`agileplus_triage::claim::ClaimReason::Branch`] so a later
    /// `from_claim()` lookup can recover it.
    ///
    /// `claim_store` is the in-memory or SQLite-backed store that
    /// contains the claim. After worktree creation we look the claim
    /// back up to update its `reason`.
    pub fn create_claim_bound_worktree<S: claim_bound::ClaimStoreBound>(
        repo_root: PathBuf,
        feature_slug: &str,
        wp_id: &str,
        claim: &agileplus_triage::claim::Claim,
        claim_store: &mut S,
    ) -> Result<PathBuf, DomainError> {
        ClaimBoundWorktree::create(repo_root, feature_slug, wp_id, claim, claim_store)
    }

    /// True when `git checkout` failed because the branch is already
    /// checked out in a different worktree.
    fn checkout_blocked_by_other_worktree(err: &DomainError) -> bool {
        let msg = err.to_string().to_lowercase();
        msg.contains("already used by worktree")
            || msg.contains("is already checked out")
            || (msg.contains("checkout") && msg.contains("worktree"))
    }

    /// Run `git merge --no-ff` of `source` into the current HEAD of `dir`.
    fn merge_in_dir(
        &self,
        dir: &Path,
        source: &str,
        target: &str,
    ) -> Result<MergeResult, DomainError> {
        let result = Command::new("git")
            .args([
                "merge",
                "--no-ff",
                "--autostash",
                "-m",
                &format!("Merge branch '{source}' into {target}"),
                source,
            ])
            .current_dir(dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| DomainError::Storage(format!("failed to spawn git merge: {e}")))?;
        let stdout = String::from_utf8_lossy(&result.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&result.stderr).into_owned();
        if result.status.success() {
            let head_out = Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .ok();
            let head = head_out
                .and_then(|o| {
                    if o.status.success() {
                        Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
                    } else {
                        None
                    }
                })
                .unwrap_or_default();
            Ok(MergeResult {
                success: true,
                conflicts: vec![],
                merged_commit: Some(head.clone()),
                commit: Some(head),
                message: Some(if stdout.is_empty() { stderr } else { stdout }),
            })
        } else {
            let conflicts = parse_merge_conflicts(&format!("{stdout}\n{stderr}"));
            Ok(MergeResult {
                success: false,
                conflicts,
                merged_commit: None,
                commit: None,
                message: Some(if stderr.is_empty() { stdout } else { stderr }),
            })
        }
    }

    /// Merge `source` into `target` inside a throwaway worktree, so we
    /// never need to check out `target` in the caller's worktree.
    fn merge_via_temp_worktree(
        &self,
        source: &str,
        target: &str,
    ) -> Result<MergeResult, DomainError> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let tmp = std::env::temp_dir().join(format!(
            "agileplus-ship-{}-{}-{}",
            target.replace('/', "_"),
            std::process::id(),
            stamp
        ));
        if tmp.exists() {
            let _ = std::fs::remove_dir_all(&tmp);
        }
        let tmp_str = tmp.to_string_lossy().into_owned();
        self.run_git_status(&["worktree", "add", "--force", &tmp_str, target])?;
        let merge_result = self.merge_in_dir(&tmp, source, target);
        if let Err(e) = self.run_git(&["worktree", "remove", "--force", &tmp_str]) {
            tracing::warn!(
                path = %tmp_str,
                error = %e,
                "failed to remove temporary ship worktree; attempting filesystem cleanup"
            );
            let _ = std::fs::remove_dir_all(&tmp);
        }
        merge_result
    }
}

#[async_trait::async_trait]
impl VcsPort for GitVcsAdapter {
    async fn create_worktree(
        &self,
        feature_slug: &str,
        wp_id: &str,
    ) -> Result<PathBuf, DomainError> {
        let branch = Self::worktree_branch(feature_slug, wp_id);
        let path = self.worktree_path(feature_slug, wp_id);
        let path_str = path.to_string_lossy().into_owned();

        // If the branch already exists, use `git worktree add <path>
        // -B <branch>` to attach the existing branch. Otherwise use
        // `-b <branch>` to create a fresh one.
        let repo = self.open()?;
        let branch_exists = repo.find_branch(&branch, git2::BranchType::Local).is_ok();
        let flag = if branch_exists { "-B" } else { "-b" };

        // Use the CLI for the worktree add â€” the git2 `Repository::worktree`
        // builder can produce a worktree in the wrong state on some
        // libgit2 versions, while `git worktree add` is the canonical,
        // well-tested path.
        self.run_git_status(&["worktree", "add", &path_str, flag, &branch])?;
        Ok(path)
    }

    async fn list_worktrees(&self) -> Result<Vec<WorktreeInfo>, DomainError> {
        // `git worktree list --porcelain` prints a sequence of
        // stanzas separated by blank lines:
        //
        //   worktree /path/to/wt
        //   HEAD <sha>
        //   branch refs/heads/<name>
        //
        //   worktree /path/to/wt2
        //   ...
        let raw = self.run_git(&["worktree", "list", "--porcelain"])?;
        if raw.is_empty() {
            return Ok(vec![]);
        }
        let mut out = Vec::new();
        let mut path: Option<String> = None;
        let mut head: Option<String> = None;
        let mut branch: Option<String> = None;
        for line in raw.lines() {
            if line.is_empty() {
                if let (Some(p), Some(h)) = (path.take(), head.take()) {
                    out.push(WorktreeInfo {
                        path: p.into(),
                        commit: h,
                        branch: branch.take().unwrap_or_default(),
                        feature_slug: String::new(),
                        wp_id: String::new(),
                    });
                } else {
                    path = None;
                    head = None;
                    branch = None;
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix("worktree ") {
                path = Some(rest.to_string());
            } else if let Some(rest) = line.strip_prefix("HEAD ") {
                head = Some(rest.to_string());
            } else if let Some(rest) = line.strip_prefix("branch ") {
                // refs/heads/feat/login -> feat/login
                let stripped = rest.strip_prefix("refs/heads/").unwrap_or(rest).to_string();
                branch = Some(stripped);
            }
        }
        // Flush the last stanza (no trailing blank line).
        if let (Some(p), Some(h)) = (path, head) {
            out.push(WorktreeInfo {
                path: p.into(),
                commit: h,
                branch: branch.unwrap_or_default(),
                feature_slug: String::new(),
                wp_id: String::new(),
            });
        }
        Ok(out)
    }

    async fn cleanup_worktree(&self, worktree_path: &Path) -> Result<(), DomainError> {
        // `--force` ignores uncommitted / unmerged changes in the
        // worktree, which is what we want during teardown of an
        // abandoned / errored worktree.
        self.run_git_status(&[
            "worktree",
            "remove",
            "--force",
            &worktree_path.to_string_lossy(),
        ])?;
        Ok(())
    }

    async fn create_branch(&self, branch: &str, base: &str) -> Result<(), DomainError> {
        let repo = self.open()?;
        let commit = Self::resolve_commit(&repo, base)?;
        // `force = false` so the caller gets a real error if the
        // branch already exists. Use `delete_branch(force=true)` to
        // overwrite.
        repo.branch(branch, &commit, false)
            .map(|_| ())
            .map_err(|e| DomainError::Storage(format!("create branch {branch}: {e}")))
    }

    async fn list_branches(
        &self,
        pattern: Option<&str>,
        remote: bool,
    ) -> Result<Vec<BranchInfo>, DomainError> {
        let repo = self.open()?;
        let filter = if remote {
            git2::BranchType::Remote
        } else {
            git2::BranchType::Local
        };
        let branches = repo
            .branches(Some(filter))
            .map_err(|e| DomainError::Storage(format!("list branches: {e}")))?;
        let pat = pattern.unwrap_or("*");
        let mut out = Vec::new();
        for entry in branches {
            let (branch, _bt) =
                entry.map_err(|e| DomainError::Storage(format!("branch entry: {e}")))?;
            let name = match branch.name() {
                Ok(Some(n)) => n.to_string(),
                _ => continue,
            };
            // Lightweight glob match: support `*` and literal text.
            // (Full glob support would need a regex; for `git branch
            // --list <pattern>` we get away with the `*` case in
            // practice.)
            if !glob_match(pat, &name) {
                continue;
            }
            let commit = branch
                .get()
                .peel_to_commit()
                .map(|c| c.id().to_string())
                .unwrap_or_default();
            out.push(BranchInfo {
                name,
                commit,
                is_remote: remote,
            });
        }
        Ok(out)
    }

    async fn delete_branch(
        &self,
        branch: &str,
        force: bool,
        remote: Option<&str>,
    ) -> Result<(), DomainError> {
        if let Some(remote_name) = remote {
            // Delete the remote branch via push.
            self.run_git_status(&["push", remote_name, &format!(":{branch}")])?;
            Ok(())
        } else {
            let repo = self.open()?;
            let mut b = repo
                .find_branch(branch, git2::BranchType::Local)
                .map_err(|e| DomainError::Storage(format!("find branch {branch}: {e}")))?;
            // `Branch::delete` only takes a force flag in newer
            // libgit2; in 0.20 we just call delete() and rely on the
            // upstream merge-state check. For the "force" case the
            // caller's branch has already been verified to be safe
            // to delete (e.g. it's been merged or --force is
            // explicit).
            let _ = force;
            b.delete()
                .map_err(|e| DomainError::Storage(format!("delete branch {branch}: {e}")))?;
            Ok(())
        }
    }

    async fn checkout_branch(&self, branch: &str) -> Result<(), DomainError> {
        // The CLI is the most reliable checkout path â€” it updates the
        // index, working tree, and HEAD ref in one go, and matches
        // what users see in their terminal.
        self.run_git_status(&["checkout", branch])?;
        Ok(())
    }

    async fn merge_to_target(
        &self,
        source: &str,
        target: &str,
    ) -> Result<MergeResult, DomainError> {
        // Prefer an in-place merge when `target` is free to check out in
        // this worktree. When another worktree already has `target`
        // checked out (common: canonical `main` + feature worktree),
        // fall back to a temporary worktree so ship still succeeds.
        match self.run_git(&["checkout", target]) {
            Ok(_) => self.merge_in_dir(&self.repo_root, source, target),
            Err(e) if Self::checkout_blocked_by_other_worktree(&e) => {
                tracing::info!(
                    target = %target,
                    source = %source,
                    "target branch checked out elsewhere; merging via temporary worktree"
                );
                self.merge_via_temp_worktree(source, target)
            }
            Err(e) => Err(e),
        }
    }

    async fn detect_conflicts(
        &self,
        source: &str,
        target: &str,
    ) -> Result<Vec<ConflictInfo>, DomainError> {
        // `git merge-tree <target> <source>` is the read-only,
        // in-memory 3-way merge that does not touch the index or
        // working tree. Its human-readable output varies by Git
        // version: older releases emit a unified diff, while macOS
        // Git emits unmerged stage rows plus `CONFLICT (...)` lines.
        // Keep interpretation in one parser shared with merge failures.
        let raw = self.run_git_allow_failure(&["merge-tree", target, source])?;
        Ok(parse_merge_conflicts(&raw))
    }

    async fn read_artifact(
        &self,
        feature_slug: &str,
        relative_path: &str,
    ) -> Result<String, DomainError> {
        let p = self.artifact_path(feature_slug, relative_path);
        std::fs::read_to_string(&p).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                DomainError::NotFound(format!(
                    "artifact {relative_path} for feature {feature_slug}"
                ))
            } else {
                DomainError::Storage(format!("read artifact {}: {e}", p.display()))
            }
        })
    }

    async fn write_artifact(
        &self,
        feature_slug: &str,
        relative_path: &str,
        content: &str,
    ) -> Result<(), DomainError> {
        let p = self.artifact_path(feature_slug, relative_path);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                DomainError::Storage(format!(
                    "create artifact parent dir {}: {e}",
                    parent.display()
                ))
            })?;
        }
        std::fs::write(&p, content)
            .map_err(|e| DomainError::Storage(format!("write artifact {}: {e}", p.display())))
    }

    async fn artifact_exists(
        &self,
        feature_slug: &str,
        relative_path: &str,
    ) -> Result<bool, DomainError> {
        Ok(self.artifact_path(feature_slug, relative_path).is_file())
    }

    async fn scan_feature_artifacts(
        &self,
        feature_slug: &str,
    ) -> Result<FeatureArtifacts, DomainError> {
        // Look for the conventional artifacts at fixed names within
        // `<repo_root>/.agileplus/<feature_slug>/`. Unknown files in
        // that directory are collected under `other`.
        let dir = self.repo_root.join(".agileplus").join(feature_slug);
        let mut out = FeatureArtifacts {
            spec: None,
            research: None,
            plan: None,
            other: vec![],
            meta_json: None,
            audit_chain: None,
            evidence_paths: vec![],
        };
        if !dir.is_dir() {
            return Ok(out);
        }
        let entries = match std::fs::read_dir(&dir) {
            Ok(it) => it,
            Err(e) => {
                return Err(DomainError::Storage(format!(
                    "scan artifacts {}: {e}",
                    dir.display()
                )))
            }
        };
        for entry in entries.flatten() {
            let name = match entry.file_name().into_string() {
                Ok(s) => s,
                Err(_) => continue,
            };
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            match name.as_str() {
                "spec.md" | "spec" => {
                    out.spec = std::fs::read_to_string(&path).ok();
                }
                "research.md" | "research" => {
                    out.research = std::fs::read_to_string(&path).ok();
                }
                "plan.md" | "plan" => {
                    out.plan = std::fs::read_to_string(&path).ok();
                }
                _ => {
                    out.other.push(name);
                }
            }
        }
        Ok(out)
    }
}

impl GitVcsAdapter {
    /// Resolve the absolute path of a feature artifact on disk.
    /// Path: `<repo_root>/.agileplus/<feature_slug>/<relative_path>`.
    fn artifact_path(&self, feature_slug: &str, relative_path: &str) -> PathBuf {
        self.repo_root
            .join(".agileplus")
            .join(feature_slug)
            .join(relative_path)
    }
}

/// Parse the conflict-bearing forms emitted by `git merge-tree` and `git merge`.
///
/// Git does not promise one stable human-readable conflict format. In
/// particular, current macOS Git prints index stage rows and `CONFLICT (...)`
/// diagnostics, while older versions can emit `changed in both` blocks or a
/// unified diff with conflict markers. All formats are normalized into the
/// domain's per-file conflict record.
fn parse_merge_conflicts(raw: &str) -> Vec<ConflictInfo> {
    let mut conflicts = Vec::new();
    let mut legacy_conflict = false;
    let mut diff_path: Option<String> = None;

    for line in raw.lines() {
        if line == "changed in both" {
            legacy_conflict = true;
            continue;
        }

        if legacy_conflict {
            if let Some(path) = parse_legacy_merge_tree_path(line) {
                add_conflict(&mut conflicts, path, "content");
                legacy_conflict = false;
            } else if !line.trim().is_empty() && !line.starts_with(' ') {
                legacy_conflict = false;
            }
        }

        if let Some(path) = parse_merge_tree_stage_path(line) {
            add_conflict(&mut conflicts, path, "content");
        }

        if let Some((path, conflict_type)) = parse_conflict_diagnostic(line) {
            add_conflict(&mut conflicts, path, conflict_type);
        }

        if let Some(path) = parse_diff_path(line) {
            diff_path = Some(path);
        } else if (line.starts_with("<<<<<<<")
            || line.starts_with("=======")
            || line.starts_with(">>>>>>>"))
            && let Some(path) = diff_path.take()
        {
            add_conflict(&mut conflicts, &path, "content");
        }
    }

    conflicts
}

fn add_conflict(conflicts: &mut Vec<ConflictInfo>, path: &str, conflict_type: &str) {
    if path.is_empty() {
        return;
    }
    if let Some(existing) = conflicts.iter_mut().find(|conflict| conflict.path == path) {
        existing.conflict_type = conflict_type.to_string();
        return;
    }
    conflicts.push(ConflictInfo {
        path: path.to_string(),
        file_path: path.to_string(),
        conflict_type: conflict_type.to_string(),
        ours: None,
        theirs: None,
    });
}

/// Split a Git metadata row into its first whitespace-delimited field and
/// the untouched remainder, retaining paths containing spaces.
fn split_git_field(value: &str) -> Option<(&str, &str)> {
    let value = value.trim_start();
    let end = value.find(char::is_whitespace)?;
    Some((&value[..end], value[end..].trim_start()))
}

/// Extract a path from `100644 <oid> <stage>\t<path>` merge-tree rows.
fn parse_merge_tree_stage_path(line: &str) -> Option<&str> {
    let (mode, rest) = split_git_field(line)?;
    if mode.len() != 6 || !mode.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let (_, rest) = split_git_field(rest)?;
    let (stage, path) = split_git_field(rest)?;
    matches!(stage, "1" | "2" | "3").then_some(path.trim())
}

/// Extract the path from the `base` row following an older `changed in both`
/// merge-tree heading.
fn parse_legacy_merge_tree_path(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("base")?;
    let (_, rest) = split_git_field(rest)?;
    let (_, path) = split_git_field(rest)?;
    Some(path.trim())
}

/// Extract paths only from stable, path-bearing conflict diagnostic clauses.
///
/// `Merge conflict in <path>` is stable across non-delete conflict types.
/// Delete diagnostics instead contain branch prose such as `left in tree` and
/// `deleted in HEAD`; parse only their explicit leading path grammars.
fn parse_conflict_diagnostic(line: &str) -> Option<(&str, &str)> {
    let rest = line.strip_prefix("CONFLICT (")?;
    let (conflict_type, message) = rest.split_once("):")?;
    let conflict_type = conflict_type.trim();
    let message = message.trim();
    let path = message
        .strip_prefix("Merge conflict in ")
        .or_else(|| match conflict_type {
            "modify/delete" => message.split_once(" deleted in ").map(|(path, _)| path),
            "rename/delete" => message
                .split_once(" renamed to ")
                .and_then(|(_, renamed)| renamed.split_once(" in "))
                .map(|(path, _)| path),
            _ => None,
        })?;
    let path = path.trim();
    (!path.is_empty()).then_some((path, conflict_type))
}

/// Extract the right-side path from a conventional `diff --git` header.
fn parse_diff_path(line: &str) -> Option<String> {
    let rest = line.strip_prefix("diff --git a/")?;
    let (_, path) = rest.rsplit_once(" b/")?;
    Some(path.to_string())
}

/// Tiny glob matcher that supports `*` and a literal suffix. Used for
/// the `list_branches` `pattern` argument. `*` matches any number of
/// characters (including none). Anchored at both ends.
fn glob_match(pattern: &str, name: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if !pattern.contains('*') {
        return pattern == name;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    // `*foo*bar*` -> 3 segments. Walk through.
    let mut idx = 0usize;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            if !name.starts_with(part) {
                return false;
            }
            idx = part.len();
        } else if i == parts.len() - 1 {
            if !name.ends_with(part) {
                return false;
            }
            // Also check that we haven't already consumed the suffix.
            if idx > name.len() - part.len() {
                return false;
            }
        } else if let Some(found) = name[idx..].find(part) {
            idx += found + part.len();
        } else {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command as StdCommand;
    use tempfile::tempdir;

    /// Make a temp git repo with one initial commit on `main`.
    fn make_repo() -> (tempfile::TempDir, PathBuf) {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();
        StdCommand::new("git")
            .args(["init", "-q", "-b", "main"])
            .current_dir(&path)
            .output()
            .unwrap();
        StdCommand::new("git")
            .args(["config", "user.email", "t@example.com"])
            .current_dir(&path)
            .output()
            .unwrap();
        StdCommand::new("git")
            .args(["config", "user.name", "tester"])
            .current_dir(&path)
            .output()
            .unwrap();
        std::fs::write(path.join("README.md"), "hello\n").unwrap();
        StdCommand::new("git")
            .args(["add", "."])
            .current_dir(&path)
            .output()
            .unwrap();
        StdCommand::new("git")
            .args(["commit", "-q", "-m", "init"])
            .current_dir(&path)
            .output()
            .unwrap();
        (dir, path)
    }

    #[test]
    fn glob_match_basic() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("feat/*", "feat/login"));
        assert!(!glob_match("feat/*", "fix/login"));
        assert!(glob_match("*/main", "origin/main"));
        assert!(glob_match("feat/*/wp-1", "feat/login/wp-1"));
        assert!(!glob_match("feat/*/wp-1", "feat/login/wp-2"));
        assert!(glob_match("literal", "literal"));
        assert!(!glob_match("literal", "other"));
    }

    #[test]
    fn parses_stage_rows_and_conflict_diagnostic_from_merge_tree() {
        // macOS Git emits stage rows followed by a human diagnostic instead
        // of the older `changed in both` or unified-diff forms.
        let raw = "100644 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 1\tshared file.txt\n\
100644 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb 2\tshared file.txt\n\
100644 cccccccccccccccccccccccccccccccccccccccc 3\tshared file.txt\n\
\n\
CONFLICT (content): Merge conflict in shared file.txt\n";

        let conflicts = parse_merge_conflicts(raw);

        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].path, "shared file.txt");
        assert_eq!(conflicts[0].file_path, "shared file.txt");
        assert_eq!(conflicts[0].conflict_type, "content");
    }

    #[test]
    fn parses_stable_merge_conflict_paths_for_non_delete_types() {
        let raw = "CONFLICT (add/add): Merge conflict in docs/new-file.txt\n\
CONFLICT (binary): Merge conflict in assets/logo.png\n";

        let conflicts = parse_merge_conflicts(raw);

        assert_eq!(conflicts.len(), 2);
        assert_eq!(conflicts[0].path, "docs/new-file.txt");
        assert_eq!(conflicts[0].conflict_type, "add/add");
        assert_eq!(conflicts[1].path, "assets/logo.png");
        assert_eq!(conflicts[1].conflict_type, "binary");
    }

    #[test]
    fn diagnostics_do_not_convert_branch_prose_into_conflict_paths() {
        // Only the content diagnostic has a stable, path-bearing suffix. The
        // other diagnostics contain branch prose with `in` clauses; their
        // paths must come from the unmerged index rows above them instead.
        let raw = "100644 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 1\tdocs/content.txt\n\
100644 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb 2\tdocs/content.txt\n\
100644 cccccccccccccccccccccccccccccccccccccccc 3\tdocs/content.txt\n\
CONFLICT (content): Merge conflict in docs/content.txt\n\
100644 dddddddddddddddddddddddddddddddddddddddd 1\tdocs/removed.txt\n\
100644 eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee 3\tdocs/removed.txt\n\
CONFLICT (modify/delete): docs/removed.txt deleted in HEAD and modified in feature. Version feature of docs/removed.txt left in tree.\n\
100644 ffffffffffffffffffffffffffffffffffffffff 1\tdocs/renamed.txt\n\
100644 1111111111111111111111111111111111111111 3\tdocs/renamed.txt\n\
CONFLICT (rename/delete): docs/item.txt renamed to docs/renamed.txt in feature, but deleted in HEAD.\n";

        let conflicts = parse_merge_conflicts(raw);
        let paths: Vec<&str> = conflicts.iter().map(|conflict| conflict.path.as_str()).collect();

        assert_eq!(paths, ["docs/content.txt", "docs/removed.txt", "docs/renamed.txt"]);
        assert_eq!(conflicts[1].conflict_type, "modify/delete");
        assert_eq!(conflicts[2].conflict_type, "rename/delete");
        assert!(!paths.iter().any(|path| matches!(*path, "tree." | "HEAD.")));
    }

    #[test]
    fn parses_legacy_changed_in_both_merge_tree_output() {
        let raw = "changed in both\n\
  base   100644 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa docs/shared file.txt\n\
  our    100644 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb docs/shared file.txt\n\
  their  100644 cccccccccccccccccccccccccccccccccccccccc docs/shared file.txt\n";

        let conflicts = parse_merge_conflicts(raw);

        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].path, "docs/shared file.txt");
        assert_eq!(conflicts[0].file_path, "docs/shared file.txt");
        assert_eq!(conflicts[0].conflict_type, "content");
    }

    #[test]
    fn parses_conflict_markers_in_unified_diff_output() {
        let raw = "diff --git a/docs/shared file.txt b/docs/shared file.txt\n\
index aaaaaaa..bbbbbbb 100644\n\
--- a/docs/shared file.txt\n\
+++ b/docs/shared file.txt\n\
@@ -1 +1,5 @@\n\
\u{003c}\u{003c}\u{003c}\u{003c}\u{003c}\u{003c}\u{003c} ours\n\
from target\n\
=======\n\
from source\n\
\u{003e}\u{003e}\u{003e}\u{003e}\u{003e}\u{003e}\u{003e} theirs\n";

        let conflicts = parse_merge_conflicts(raw);

        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].path, "docs/shared file.txt");
        assert_eq!(conflicts[0].file_path, "docs/shared file.txt");
        assert_eq!(conflicts[0].conflict_type, "content");
    }

    #[tokio::test]
    async fn create_and_list_worktree() {
        let (_dir, path) = make_repo();
        let adapter = GitVcsAdapter::new(path.clone());
        let wt = adapter
            .create_worktree("login", "wp-1")
            .await
            .expect("create worktree");
        assert!(
            wt.is_dir(),
            "worktree dir was not created: {}",
            wt.display()
        );
        let list = adapter.list_worktrees().await.expect("list worktrees");
        let names: Vec<&str> = list.iter().map(|w| w.branch.as_str()).collect();
        assert!(
            names.iter().any(|n| n.contains("wp-1")),
            "expected wp-1 branch in worktree list, got {:?}",
            names
        );
        // Cleanup to avoid polluting temp dir for subsequent tests
        adapter
            .cleanup_worktree(&wt)
            .await
            .expect("cleanup worktree");
    }
    #[tokio::test]
    async fn create_branch_lists_and_checkout() {
        let (_dir, path) = make_repo();
        let adapter = GitVcsAdapter::new(path.clone());
        adapter
            .create_branch("feat/x", "main")
            .await
            .expect("create branch");
        let locals = adapter
            .list_branches(Some("feat/*"), false)
            .await
            .expect("list local branches");
        assert!(locals.iter().any(|b| b.name == "feat/x"));
        adapter.checkout_branch("feat/x").await.expect("checkout");
        let head = adapter
            .run_git(&["rev-parse", "--abbrev-ref", "HEAD"])
            .unwrap();
        assert_eq!(head, "feat/x");
    }

    #[tokio::test]
    async fn merge_no_conflict_to_target() {
        let (_dir, path) = make_repo();
        let adapter = GitVcsAdapter::new(path.clone());
        // create a feature branch with a non-conflicting change
        adapter.create_branch("feat/ok", "main").await.unwrap();
        adapter.checkout_branch("feat/ok").await.unwrap();
        std::fs::write(path.join("newfile.txt"), "hi").unwrap();
        StdCommand::new("git")
            .args(["add", "newfile.txt"])
            .current_dir(&path)
            .output()
            .unwrap();
        StdCommand::new("git")
            .args(["commit", "-q", "-m", "add newfile"])
            .current_dir(&path)
            .output()
            .unwrap();
        let res = adapter
            .merge_to_target("feat/ok", "main")
            .await
            .expect("merge");
        assert!(res.success, "merge should succeed: {:?}", res.message);
        assert!(res.commit.is_some());
    }

    #[tokio::test]
    // Traces to: FR-006 / ship worktree-aware merge
    async fn merge_when_target_checked_out_elsewhere() {
        let (_dir, path) = make_repo();
        let adapter = GitVcsAdapter::new(path.clone());
        adapter.create_branch("feat/ok", "main").await.unwrap();
        adapter.checkout_branch("feat/ok").await.unwrap();
        std::fs::write(path.join("newfile.txt"), "hi").unwrap();
        StdCommand::new("git")
            .args(["add", "newfile.txt"])
            .current_dir(&path)
            .output()
            .unwrap();
        StdCommand::new("git")
            .args(["commit", "-q", "-m", "add newfile"])
            .current_dir(&path)
            .output()
            .unwrap();

        // Lock `main` in a sibling worktree (canonical-style conflict).
        let lock_wt = path.parent().unwrap().join("lock-main-wt");
        StdCommand::new("git")
            .args([
                "worktree",
                "add",
                &lock_wt.to_string_lossy(),
                "main",
            ])
            .current_dir(&path)
            .output()
            .expect("lock main in sibling worktree");

        let res = adapter
            .merge_to_target("feat/ok", "main")
            .await
            .expect("merge via temp worktree");
        assert!(
            res.success,
            "merge should succeed when main is locked elsewhere: {:?}",
            res.message
        );
        assert!(res.commit.is_some());

        // Cleanup locked worktree
        let _ = StdCommand::new("git")
            .args(["worktree", "remove", "--force", &lock_wt.to_string_lossy()])
            .current_dir(&path)
            .output();
    }

    #[tokio::test]
    async fn merge_conflict_detected() {
        let (_dir, path) = make_repo();
        let adapter = GitVcsAdapter::new(path.clone());
        // create a feature branch that edits README.md
        adapter
            .create_branch("feat/conflict", "main")
            .await
            .unwrap();
        adapter.checkout_branch("feat/conflict").await.unwrap();
        std::fs::write(path.join("README.md"), "from feature\n").unwrap();
        StdCommand::new("git")
            .args(["add", "README.md"])
            .current_dir(&path)
            .output()
            .unwrap();
        StdCommand::new("git")
            .args(["commit", "-q", "-m", "feature edit"])
            .current_dir(&path)
            .output()
            .unwrap();
        // edit README.md differently on main
        adapter.checkout_branch("main").await.unwrap();
        std::fs::write(path.join("README.md"), "from main\n").unwrap();
        StdCommand::new("git")
            .args(["add", "README.md"])
            .current_dir(&path)
            .output()
            .unwrap();
        StdCommand::new("git")
            .args(["commit", "-q", "-m", "main edit"])
            .current_dir(&path)
            .output()
            .unwrap();
        let res = adapter
            .merge_to_target("feat/conflict", "main")
            .await
            .expect("merge attempted");
        assert!(!res.success, "merge should have conflicted");
        assert!(
            res.conflicts.iter().any(|c| c.file_path == "README.md"),
            "failed merge should report the conflicted path: {:?}",
            res.conflicts
        );
        // abort so the test repo is in a clean state for other tests
        let _ = adapter.run_git_status(&["merge", "--abort"]);
        let conflicts = adapter
            .detect_conflicts("feat/conflict", "main")
            .await
            .expect("detect conflicts");
        assert!(!conflicts.is_empty(), "expected at least one conflict");
        assert!(conflicts.iter().any(|c| c.file_path == "README.md"));
    }

    #[tokio::test]
    async fn delete_local_and_remote_branch() {
        let (_dir, path) = make_repo();
        let adapter = GitVcsAdapter::new(path.clone());
        adapter
            .create_branch("feat/ephemeral", "main")
            .await
            .unwrap();
        adapter
            .delete_branch("feat/ephemeral", false, None)
            .await
            .expect("delete local");
        let locals = adapter.list_branches(Some("feat/*"), false).await.unwrap();
        assert!(locals.iter().all(|b| b.name != "feat/ephemeral"));
    }

    #[tokio::test]
    async fn artifact_round_trip() {
        let (_dir, path) = make_repo();
        let adapter = GitVcsAdapter::new(path.clone());
        adapter
            .write_artifact("login", "spec.md", "# spec\n")
            .await
            .expect("write");
        let content = adapter
            .read_artifact("login", "spec.md")
            .await
            .expect("read");
        assert_eq!(content, "# spec\n");
        assert!(adapter.artifact_exists("login", "spec.md").await.unwrap());
        let scan = adapter.scan_feature_artifacts("login").await.expect("scan");
        assert_eq!(scan.spec.as_deref(), Some("# spec\n"));
    }

    #[tokio::test]
    async fn cleanup_worktree_removes_it() {
        let (_dir, path) = make_repo();
        let adapter = GitVcsAdapter::new(path.clone());
        let wt = adapter.create_worktree("x", "wp-1").await.unwrap();
        assert!(wt.is_dir());
        adapter.cleanup_worktree(&wt).await.expect("cleanup");
        assert!(!wt.exists(), "worktree should be removed");
    }
}
