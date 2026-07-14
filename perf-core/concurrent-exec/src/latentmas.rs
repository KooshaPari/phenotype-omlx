//! LatentMAS concurrent adapter — multiple latent agents in parallel.

use crate::{AgentId, ExecBackend, ExecRequest, ExecResult, JobError};
use async_trait::async_trait;
use std::sync::Arc;

/// LatentMAS in-process backend. Spawns N agents on the same prompt in parallel
/// and merges their hidden-state trajectories through a configurable reducer.
pub struct LatentMasBackend {
    pub n_agents: usize,
    pub device: Arc<str>,
    pub merge: Arc<MergeFn>,
}

pub type MergeFn = dyn Fn(&[ExecResult]) -> ExecResult + Send + Sync;

impl LatentMasBackend {
    pub fn new(n_agents: usize, device: impl Into<Arc<str>>) -> Self {
        Self {
            n_agents,
            device: device.into(),
            merge: Arc::new(|rs| match rs.first() {
                Some(r) => r.clone(),
                None => ExecResult::empty(),
            }),
        }
    }
}

#[async_trait]
impl ExecBackend for LatentMasBackend {
    async fn run(
        &self,
        id: AgentId,
        req: ExecRequest,
    ) -> Result<ExecResult, JobError> {
        // In real LatentMAS, this would:
        //   1. run each latent agent's forward pass under MPS/CPU
        //   2. merge their hidden states through the configured reducer
        // The stub returns a deterministically-merged ExecResult so callers can
        // wire scheduling without depending on a full model.
        let merged = ExecResult::ok_with_text(format!(
            "[latentmas:{} n={} device={} chars={}]",
            id,
            self.n_agents,
            self.device,
            req.prompt.len(),
        ));
        Ok(merged)
    }
}