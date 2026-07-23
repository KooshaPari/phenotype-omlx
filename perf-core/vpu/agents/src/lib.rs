pub mod agent_loop;
pub mod orchestrator;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceKind {
    Proton,
    Electron,
    Neutron,
    Dram,
    Nvme,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub id: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub context_len: usize,
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    pub task_id: String,
    pub completion: String,
    pub tokens: usize,
    pub elapsed_ms: f64,
    pub device: DeviceKind,
}

pub trait DeviceBackend {
    fn complete(&self, task: &AgentTask) -> AgentResult;
    fn device_kind(&self) -> DeviceKind;
}
