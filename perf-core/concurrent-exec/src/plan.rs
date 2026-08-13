//! Concurrent execution plan for LatentMAS / TiDAR / JetSpec / SSD.

use crate::{ExecBackend, ExecRequest, ExecResult, GovernorConfig, JobError, ResourceGovernor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Semaphore};
use tokio::time::timeout;

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
    pub queue_capacity: usize,
    pub max_fanout: usize,
    pub job_timeout: Duration,
    pub governor: ResourceGovernor,
    pub backends: HashMap<AgentId, Arc<dyn ExecBackend>>,
    tx: mpsc::Sender<Job>,
}

impl Scheduler {
    pub fn new(strategy: ScheduleStrategy, max_concurrency: usize) -> (Self, JobReceiver) {
        let concurrency = max_concurrency.max(1);
        Self::with_limits(
            strategy,
            concurrency,
            concurrency.saturating_mul(4).max(1),
            concurrency.saturating_mul(2).max(1),
            Duration::from_secs(30),
        )
    }

    pub fn with_limits(
        strategy: ScheduleStrategy,
        max_concurrency: usize,
        queue_capacity: usize,
        max_fanout: usize,
        job_timeout: Duration,
    ) -> (Self, JobReceiver) {
        let queue_capacity = queue_capacity.max(1);
        let max_fanout = max_fanout.max(1);
        let (tx, rx) = mpsc::channel(queue_capacity);
        let mut governor_config = GovernorConfig::for_concurrency(max_concurrency);
        governor_config.max_queue = queue_capacity;
        governor_config.acquire_timeout = job_timeout.max(Duration::from_millis(1));
        let s = Self {
            strategy,
            max_concurrency: Arc::new(Semaphore::new(max_concurrency.max(1))),
            queue_capacity,
            max_fanout,
            job_timeout: job_timeout.max(Duration::from_millis(1)),
            governor: ResourceGovernor::new(governor_config),
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
        self.tx.try_send(job).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => JobError::QueueFull,
            mpsc::error::TrySendError::Closed(_) => {
                JobError::Backend("scheduler channel closed".into())
            }
        })
    }
}

/// Pull-based job receiver.
pub struct JobReceiver {
    rx: mpsc::Receiver<Job>,
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
    if scheduler.backends.len() > scheduler.max_fanout {
        return Err(JobError::FanoutLimit(scheduler.max_fanout));
    }
    let mut handles = Vec::new();
    for (id, backend) in &scheduler.backends {
        let id = id.clone();
        let backend = backend.clone();
        let permit = scheduler
            .governor
            .acquire(&payload)
            .await
            .map_err(|error| JobError::Backend(error.to_string()))?;
        let p = payload.clone();
        let deadline = scheduler.job_timeout;
        let h = tokio::spawn(async move {
            let _p = permit;
            timeout(deadline, backend.run(id, p))
                .await
                .map_err(|_| JobError::Timeout)?
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

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::Arc;
    use tokio::time::sleep;

    struct SlowBackend;

    #[async_trait]
    impl ExecBackend for SlowBackend {
        async fn run(&self, _id: AgentId, _req: ExecRequest) -> Result<ExecResult, JobError> {
            sleep(Duration::from_millis(20)).await;
            Ok(ExecResult::empty())
        }
    }

    fn request() -> ExecRequest {
        ExecRequest {
            prompt: "bounded".into(),
            max_tokens: 1,
            temperature: 0.0,
            stop: Vec::new(),
        }
    }

    #[test]
    fn dispatch_rejects_queue_overflow() {
        let (scheduler, _receiver) = Scheduler::with_limits(
            ScheduleStrategy::RoundRobin,
            1,
            1,
            1,
            Duration::from_secs(1),
        );
        let job = || Job {
            id: 1,
            agent: AgentId::new("a"),
            payload: request(),
        };
        assert!(scheduler.dispatch(job()).is_ok());
        assert!(matches!(
            scheduler.dispatch(job()),
            Err(JobError::QueueFull)
        ));
    }

    #[tokio::test]
    async fn fan_out_enforces_timeout_and_fanout_cap() {
        let (mut scheduler, _receiver) =
            Scheduler::with_limits(ScheduleStrategy::FanOut, 1, 2, 1, Duration::from_millis(1));
        scheduler = scheduler.register(AgentId::new("a"), Arc::new(SlowBackend));
        let result = fan_out(&scheduler, request()).await;
        assert!(matches!(result, Err(JobError::Timeout)));

        let (scheduler, _receiver) =
            Scheduler::with_limits(ScheduleStrategy::FanOut, 2, 2, 1, Duration::from_secs(1));
        let scheduler = scheduler
            .register(AgentId::new("a"), Arc::new(SlowBackend))
            .register(AgentId::new("b"), Arc::new(SlowBackend));
        assert!(matches!(
            fan_out(&scheduler, request()).await,
            Err(JobError::FanoutLimit(1))
        ));
    }
}
