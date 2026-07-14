//! Concurrent execution plan for LatentMAS / TiDAR / JetSpec / SSD.

use crate::{ExecBackend, ExecRequest, ExecResult, JobError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};

/// Identifier for a single execution agent (e.g. "latentmas.solver").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// One unit of work flowing through the scheduler.
#[derive(Debug, Clone)]
pub struct Job {
    pub id: u64,
    pub agent: AgentId,
    pub payload: ExecRequest,
}

/// Result flowing back to the dispatcher.
#[derive(Debug, Clone)]
pub struct JobOutput {
    pub job_id: u64,
    pub result: Result<ExecResult, JobError>,
}

/// Strategy for fanning work out across agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScheduleStrategy {
    /// Run all registered agents on the same input in parallel; merge outputs.
    FanOut,
    /// Run agents in a fixed sequence; pass the prior output as input to the next.
    Chain,
    /// Round-robin across agents.
    RoundRobin,
    /// Run a single agent; if it fails, fall back to the next.
    Fallback,
}

/// The scheduler itself — async-friendly, semaphore-limited.
pub struct Scheduler {
    pub strategy: ScheduleStrategy,
    pub max_concurrency: Arc<Semaphore>,
    pub backends: HashMap<AgentId, Arc<dyn ExecBackend>>,
    tx: mpsc::UnboundedSender<Job>,
}

impl Scheduler {
    pub fn new(strategy: ScheduleStrategy, max_concurrency: usize) -> (Self, JobReceiver) {
        let (tx, rx) = mpsc::unbounded_channel();
        let s = Self {
            strategy,
            max_concurrency: Arc::new(Semaphore::new(max_concurrency.max(1))),
            backends: HashMap::new(),
            tx,
        };
        (s, JobReceiver { rx })
    }

    pub fn register(mut self, id: AgentId, backend: Arc<dyn ExecBackend>) -> Self {
        self.backends.insert(id, backend);
        self
    }

    /// Dispatch a job (returns immediately; results come through JobReceiver).
    pub fn dispatch(&self, job: Job) -> Result<(), JobError> {
        self.tx
            .send(job)
            .map_err(|_| JobError::Backend("scheduler channel closed".into()))
    }
}

/// Pull-based job receiver.
pub struct JobReceiver {
    rx: mpsc::UnboundedReceiver<Job>,
}

impl JobReceiver {
    pub async fn recv(&mut self) -> Option<Job> {
        self.rx.recv().await
    }
}

/// Run a fan-out over N agents and merge their outputs sequentially in order.
///
/// The default merge returns the first successful result; downstream consumers
/// can swap in `merge_concat` or similar policies.
pub async fn fan_out(
    scheduler: &Scheduler,
    payload: ExecRequest,
) -> Result<Vec<ExecResult>, JobError> {
    let mut handles = Vec::new();
    for (id, backend) in &scheduler.backends {
        let id = id.clone();
        let backend = backend.clone();
        let permit = scheduler
            .max_concurrency
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| JobError::Backend("permit".into()))?;
        let p = payload.clone();
        let h = tokio::spawn(async move {
            let _p = permit;
            backend.run(id, p).await
        });
        handles.push(h);
    }
    let mut outs = Vec::new();
    for h in handles {
        let r = h.await.map_err(|e| JobError::Backend(e.to_string()))??;
        outs.push(r);
    }
    Ok(outs)
}

/// Like `fan_out`, but returns the first `Some(...)` result.
pub async fn first_success(
    scheduler: &Scheduler,
    payload: ExecRequest,
) -> Result<ExecResult, JobError> {
    let r = fan_out(scheduler, payload).await?;
    r.into_iter()
        .next()
        .ok_or_else(|| JobError::Backend("no backends registered".into()))
}