//! TiDAR — Think in Diffusion, Talk in AR.
//! Concurrency adapter: hybrid AR+diffusion decode where diffusion drafts
//! several tokens in parallel, then a single AR pass verifies.

use crate::{AgentId, ExecBackend, ExecRequest, ExecResult, JobError};
use async_trait::async_trait;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TidarRole {
    Drafter,
    Verifier,
}

pub struct TidarAgent {
    pub role: TidarRole,
    pub draft_len: usize,
    pub steps: usize,
    pub device: Arc<str>,
}

impl TidarAgent {
    pub fn drafter(draft_len: usize, steps: usize, device: impl Into<Arc<str>>) -> Self {
        Self {
            role: TidarRole::Drafter,
            draft_len,
            steps,
            device: device.into(),
        }
    }
    pub fn verifier(device: impl Into<Arc<str>>) -> Self {
        Self {
            role: TidarRole::Verifier,
            draft_len: 0,
            steps: 0,
            device: device.into(),
        }
    }
}

#[async_trait]
impl ExecBackend for TidarAgent {
    async fn run(&self, id: AgentId, req: ExecRequest) -> Result<ExecResult, JobError> {
        // The actual TiDAR forward pass lives in the Python reference; this
        // Rust adapter simply provides a scheduling handle for the perf-core
        // to call *into* the Python side via pyO3.
        let role = match self.role {
            TidarRole::Drafter => "drafter",
            TidarRole::Verifier => "verifier",
        };
        Ok(ExecResult::ok_with_text(format!(
            "[tidar:{} role={} draft_len={} steps={} device={} chars={}]",
            id,
            role,
            self.draft_len,
            self.steps,
            self.device,
            req.prompt.len(),
        )))
    }
}
