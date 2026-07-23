use crate::{AgentResult, AgentTask, DeviceBackend, DeviceKind};

pub struct Orchestrator {
    backends: Vec<Box<dyn DeviceBackend + Send + Sync>>,
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl Orchestrator {
    pub fn new() -> Self {
        Self { backends: vec![] }
    }
    pub fn add_backend(&mut self, backend: Box<dyn DeviceBackend + Send + Sync>) {
        self.backends.push(backend);
    }
    pub fn dispatch(&self, task: &AgentTask) -> AgentResult {
        self.backends
            .first()
            .map(|b| b.complete(task))
            .unwrap_or_else(|| AgentResult {
                task_id: task.id.clone(),
                completion: String::new(),
                tokens: 0,
                elapsed_ms: 0.0,
                device: DeviceKind::Dram,
            })
    }
    pub fn dispatch_all(&self, task: &AgentTask) -> Vec<AgentResult> {
        self.backends.iter().map(|b| b.complete(task)).collect()
    }
}
