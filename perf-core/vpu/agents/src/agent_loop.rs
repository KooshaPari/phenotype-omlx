use crate::{AgentResult, AgentTask, DeviceBackend};

pub struct AgentLoop;

impl AgentLoop {
    pub fn run_code_gen(backend: &dyn DeviceBackend, spec: &str, max_tokens: usize, max_iterations: usize) -> Vec<AgentResult> {
        (0..max_iterations).map(|_| backend.complete(&AgentTask { id: "code-gen".into(), prompt: spec.into(), max_tokens, context_len: 0, temperature: 0.7 })).collect()
    }
}
