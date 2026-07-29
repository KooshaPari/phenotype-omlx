//! Concurrent execution — fan-out / chain / fallback scheduling over
//! heterogeneous execution backends (LatentMAS, TiDAR, JetSpec, SSD, …).

pub mod jetspec;
pub mod latentmas;
pub mod plan;
pub mod ssd;
pub mod tidar;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum JobError {
    #[error("backend error: {0}")]
    Backend(String),
    #[error("timeout")]
    Timeout,
    #[error("cancelled")]
    Cancelled,
    #[error("scheduler queue is full")]
    QueueFull,
    #[error("fan-out exceeds scheduler limit ({0})")]
    FanoutLimit(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecResult {
    pub text: String,
    pub tokens: usize,
    pub elapsed_ms: u64,
}

impl ExecResult {
    pub fn empty() -> Self {
        Self {
            text: String::new(),
            tokens: 0,
            elapsed_ms: 0,
        }
    }
    pub fn ok_with_text(s: impl Into<String>) -> Self {
        Self {
            text: s.into(),
            tokens: 1,
            elapsed_ms: 1,
        }
    }
}

#[async_trait]
pub trait ExecBackend: Send + Sync {
    async fn run(&self, id: plan::AgentId, req: ExecRequest) -> Result<ExecResult, JobError>;
}

pub use plan::{fan_out, first_success, AgentId, Job, JobOutput, ScheduleStrategy, Scheduler};
