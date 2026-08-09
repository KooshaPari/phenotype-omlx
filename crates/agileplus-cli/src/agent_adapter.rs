//! Real agent adapter for dispatching to configured backends.
//!
//! Supports multiple agent backends:
//! - `claude-cli`: Claude Code CLI (primary)
//! - `cheap-llm-mcp`: Cheap LLM via MCP (fallback)
//! - `stub`: No-op stub (testing only)
//!
//! Backend selection via environment variable `FOCAL_AGENT_BACKEND` (default: claude-cli).

use std::env;
use std::process::Stdio;

use agileplus_domain::error::DomainError;
use agileplus_domain::ports::agent::{AgentConfig, AgentPort, AgentResult, AgentStatus, AgentTask};
use dashmap::DashMap;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use uuid::Uuid;

/// Real agent adapter with pluggable backend support.
pub struct RealAgentAdapter {
    backend: AgentBackend,
    jobs: std::sync::Arc<DashMap<String, JobState>>,
}

#[derive(Debug, Clone)]
enum AgentBackend {
    ClaudeCli,
    CheapLlmMcp,
    Stub,
}

impl AgentBackend {
    fn from_env() -> Self {
        Self::from_backend_name(env::var("FOCAL_AGENT_BACKEND").ok().as_deref())
    }

    fn from_backend_name(name: Option<&str>) -> Self {
        match name {
            Some("cheap-llm-mcp") => AgentBackend::CheapLlmMcp,
            Some("stub") => AgentBackend::Stub,
            _ => AgentBackend::ClaudeCli, // default
        }
    }
}

#[derive(Debug, Clone)]
struct JobState {
    status: AgentStatus,
}

impl RealAgentAdapter {
    /// Create a new real agent adapter, selecting backend from FOCAL_AGENT_BACKEND env var.
    pub fn new() -> Self {
        let backend = AgentBackend::from_env();
        tracing::info!("Initializing agent adapter with backend: {:?}", backend);
        Self {
            backend,
            jobs: std::sync::Arc::new(DashMap::new()),
        }
    }
}

impl Default for RealAgentAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentPort for RealAgentAdapter {
    /// Synchronous dispatch: spawn agent, wait for completion, return result.
    async fn dispatch(
        &self,
        task: AgentTask,
        config: &AgentConfig,
    ) -> Result<AgentResult, DomainError> {
        match self.backend {
            AgentBackend::ClaudeCli => dispatch_claude_cli(&task, config).await,
            AgentBackend::CheapLlmMcp => dispatch_cheap_llm_mcp(&task, config).await,
            AgentBackend::Stub => dispatch_stub(&task, config).await,
        }
    }

    /// Asynchronous dispatch: spawn agent in background, return job ID.
    async fn dispatch_async(
        &self,
        task: AgentTask,
        config: &AgentConfig,
    ) -> Result<String, DomainError> {
        let job_id = Uuid::new_v4().to_string();
        let result = self.dispatch(task.clone(), config).await?;
        let status = if result.success {
            AgentStatus::Completed { result }
        } else {
            AgentStatus::Failed {
                error: result.stderr.clone(),
            }
        };
        self.jobs.insert(job_id.clone(), JobState { status });
        Ok(job_id)
    }

    /// Query the status of a dispatched agent job.
    async fn query_status(&self, job_id: &str) -> Result<AgentStatus, DomainError> {
        self.jobs
            .get(job_id)
            .map(|entry| entry.status.clone())
            .ok_or_else(|| DomainError::NotFound(format!("job {} not found", job_id)))
    }

    /// Cancel a running agent job.
    async fn cancel(&self, job_id: &str) -> Result<(), DomainError> {
        self.jobs.remove(job_id);
        Ok(())
    }

    /// Send an instruction (fix request, feedback) to the running agent.
    async fn send_instruction(&self, job_id: &str, instruction: &str) -> Result<(), DomainError> {
        // For now, this is a no-op. In production, this would write to the agent's
        // stdin or create an instruction file in the worktree for the agent to consume.
        tracing::info!("Instruction for {}: {}", job_id, instruction);
        Ok(())
    }
}

/// Dispatch via Claude Code CLI with --print mode.
async fn dispatch_claude_cli(
    task: &AgentTask,
    config: &AgentConfig,
) -> Result<AgentResult, DomainError> {
    // Read the prompt file
    let prompt_content = tokio::fs::read_to_string(&task.prompt_path)
        .await
        .map_err(|e| DomainError::Agent(format!("reading prompt: {}", e)))?;

    // Build command: claude --print <prompt>
    let mut cmd = Command::new("claude");
    cmd.arg("--print")
        .arg("--dangerously-skip-permissions") // allow non-interactive use
        .current_dir(&task.worktree_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Add extra args from config
    for arg in &config.extra_args {
        cmd.arg(arg);
    }

    // Spawn the process
    let mut child = cmd
        .spawn()
        .map_err(|e| DomainError::Agent(format!("spawn claude: {}", e)))?;

    // Write prompt to stdin
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(prompt_content.as_bytes())
            .await
            .map_err(|e| DomainError::Agent(format!("write to claude stdin: {}", e)))?;
    }

    // Wait with timeout
    let output = tokio::time::timeout(
        std::time::Duration::from_secs(config.timeout_secs),
        child.wait_with_output(),
    )
    .await
    .map_err(|_| DomainError::Timeout(config.timeout_secs))?
    .map_err(|e| DomainError::Agent(format!("wait for claude: {}", e)))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let pr_url = extract_pr_url(&stdout);
    let commits = extract_commits(&stdout);

    Ok(AgentResult {
        success: output.status.success(),
        pr_url,
        commits,
        stdout,
        stderr,
        exit_code: output.status.code().unwrap_or(-1),
    })
}

/// Dispatch via cheap-llm-mcp backend.
async fn dispatch_cheap_llm_mcp(
    task: &AgentTask,
    config: &AgentConfig,
) -> Result<AgentResult, DomainError> {
    // Similar to Claude CLI but with cheap-llm-mcp CLI interface
    let prompt_content = tokio::fs::read_to_string(&task.prompt_path)
        .await
        .map_err(|e| DomainError::Agent(format!("reading prompt: {}", e)))?;

    let mut cmd = Command::new("cheap-llm-mcp");
    cmd.arg("dispatch")
        .arg("--model")
        .arg("minimax") // configurable via env var in production
        .current_dir(&task.worktree_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    for arg in &config.extra_args {
        cmd.arg(arg);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| DomainError::Agent(format!("spawn cheap-llm: {}", e)))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(prompt_content.as_bytes())
            .await
            .map_err(|e| DomainError::Agent(format!("write to cheap-llm stdin: {}", e)))?;
    }

    let output = tokio::time::timeout(
        std::time::Duration::from_secs(config.timeout_secs),
        child.wait_with_output(),
    )
    .await
    .map_err(|_| DomainError::Timeout(config.timeout_secs))?
    .map_err(|e| DomainError::Agent(format!("wait for cheap-llm: {}", e)))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let pr_url = extract_pr_url(&stdout);
    let commits = extract_commits(&stdout);

    Ok(AgentResult {
        success: output.status.success(),
        pr_url,
        commits,
        stdout,
        stderr,
        exit_code: output.status.code().unwrap_or(-1),
    })
}

/// Dispatch to stub backend (no-op, for testing).
async fn dispatch_stub(
    task: &AgentTask,
    _config: &AgentConfig,
) -> Result<AgentResult, DomainError> {
    tracing::warn!(
        "Using stub agent backend for WP {} in {}",
        task.wp_id,
        task.worktree_path.display()
    );
    Ok(AgentResult {
        success: true,
        pr_url: None,
        commits: vec![],
        stdout: format!("[stub] dispatched WP {}", task.wp_id),
        stderr: String::new(),
        exit_code: 0,
    })
}

/// Extract PR URL from agent stdout using regex.
fn extract_pr_url(stdout: &str) -> Option<String> {
    // Look for GitHub PR URLs: https://github.com/owner/repo/pull/NNN
    let re = regex::Regex::new(r"https://github\.com/[^/]+/[^/]+/pull/\d+").ok()?;
    re.find(stdout).map(|m| m.as_str().to_string())
}

/// Extract commit SHAs from agent stdout.
fn extract_commits(stdout: &str) -> Vec<String> {
    // Look for commit SHAs (40-char hex strings or abbreviated)
    let re = regex::Regex::new(r"\b([0-9a-f]{7,40})\b").ok();
    re.as_ref()
        .map(|r| {
            r.captures_iter(stdout)
                .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Traces to: FR-013-dispatch-backend-selection
    fn test_backend_selection_from_env() {
        assert!(matches!(
            AgentBackend::from_backend_name(None),
            AgentBackend::ClaudeCli
        ));
        assert!(matches!(
            AgentBackend::from_backend_name(Some("cheap-llm-mcp")),
            AgentBackend::CheapLlmMcp
        ));
        assert!(matches!(
            AgentBackend::from_backend_name(Some("stub")),
            AgentBackend::Stub
        ));
    }

    #[test]
    // Traces to: FR-013-pr-url-extraction
    fn test_pr_url_extraction() {
        let stdout = "Successfully created PR at https://github.com/example/repo/pull/42";
        let pr_url = extract_pr_url(stdout);
        assert_eq!(
            pr_url,
            Some("https://github.com/example/repo/pull/42".to_string())
        );

        let no_pr = "No PR found";
        assert_eq!(extract_pr_url(no_pr), None);
    }

    #[test]
    // Traces to: FR-013-commit-extraction
    fn test_commit_extraction() {
        let stdout = "Committed 1234567 and a1b2c3d with message";
        let commits = extract_commits(stdout);
        assert_eq!(commits.len(), 2);
        assert!(commits.contains(&"1234567".to_string()));
        assert!(commits.contains(&"a1b2c3d".to_string()));
    }

    #[tokio::test]
    // Traces to: FR-013-dispatch-stub-backend
    async fn test_stub_dispatch() {
        let adapter = RealAgentAdapter {
            backend: AgentBackend::Stub,
            jobs: std::sync::Arc::new(DashMap::new()),
        };

        let task = AgentTask {
            wp_id: "WP08".to_string(),
            feature_slug: "test".to_string(),
            prompt_path: "/tmp/prompt.md".into(),
            worktree_path: "/tmp/test-wt".into(),
            context_files: vec![],
        };

        let config = AgentConfig {
            kind: agileplus_domain::ports::agent::AgentKind::ClaudeCode,
            max_review_cycles: 3,
            timeout_secs: 300,
            extra_args: vec![],
        };

        let result = adapter.dispatch(task, &config).await;
        assert!(result.is_ok());
        assert!(result.unwrap().success);
    }

    #[tokio::test]
    // Traces to: FR-013-job-status-tracking
    async fn test_job_status_tracking() {
        // Create temporary prompt file
        let temp_dir = tempfile::tempdir().unwrap();
        let prompt_path = temp_dir.path().join("prompt.md");
        std::fs::write(&prompt_path, "Test prompt").unwrap();

        // Create adapter with stub backend directly to avoid env var issues
        let adapter = RealAgentAdapter {
            backend: AgentBackend::Stub,
            jobs: std::sync::Arc::new(DashMap::new()),
        };

        let task = AgentTask {
            wp_id: "WP01".to_string(),
            feature_slug: "test".to_string(),
            prompt_path,
            worktree_path: temp_dir.path().to_path_buf(),
            context_files: vec![],
        };

        let config = AgentConfig {
            kind: agileplus_domain::ports::agent::AgentKind::ClaudeCode,
            max_review_cycles: 3,
            timeout_secs: 300,
            extra_args: vec![],
        };

        let job_id = adapter.dispatch_async(task.clone(), &config).await.unwrap();

        let status = adapter.query_status(&job_id).await;
        assert!(status.is_ok());

        let cancel_result = adapter.cancel(&job_id).await;
        assert!(cancel_result.is_ok());
    }
}
