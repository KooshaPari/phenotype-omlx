//! JetSpec — Tree-attention speculative decoding.

use crate::{AgentId, ExecBackend, ExecRequest, ExecResult, JobError};
use async_trait::async_trait;
use std::sync::Arc;

pub struct JetSpecBackend {
    pub tree_width: usize,
    pub tree_depth: usize,
    pub device: Arc<str>,
}

impl JetSpecBackend {
    pub fn new(tree_width: usize, tree_depth: usize, device: impl Into<Arc<str>>) -> Self {
        Self {
            tree_width,
            tree_depth,
            device: device.into(),
        }
    }
}

#[async_trait]
impl ExecBackend for JetSpecBackend {
    async fn run(&self, id: AgentId, req: ExecRequest) -> Result<ExecResult, JobError> {
        Ok(ExecResult::ok_with_text(format!(
            "[jetspec:{} w={} d={} device={} chars={}]",
            id,
            self.tree_width,
            self.tree_depth,
            self.device,
            req.prompt.len(),
        )))
    }
}
