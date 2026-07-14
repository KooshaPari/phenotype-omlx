//! SSD — Self-speculative decoding. Same model, prompt-lookup draft.

use crate::{AgentId, ExecBackend, ExecRequest, ExecResult, JobError};
use async_trait::async_trait;
use std::sync::Arc;

pub struct SsdBackend {
    pub gamma: usize,
    pub device: Arc<str>,
}

impl SsdBackend {
    pub fn new(gamma: usize, device: impl Into<Arc<str>>) -> Self {
        Self { gamma, device: device.into() }
    }
}

#[async_trait]
impl ExecBackend for SsdBackend {
    async fn run(
        &self,
        id: AgentId,
        req: ExecRequest,
    ) -> Result<ExecResult, JobError> {
        Ok(ExecResult::ok_with_text(format!(
            "[ssd:{} gamma={} device={} chars={}]",
            id, self.gamma, self.device, req.prompt.len(),
        )))
    }
}