pub mod mmlu;
pub mod gpqa;
pub mod terminal_bench;
pub mod perplexity;
pub mod fixture_backend;

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TaskSpec {
    pub id: String,
    pub suite: Suite,
    pub prompt: String,
    pub expected: Option<String>,
    pub choices: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Suite {
    MMLU,
    GPQA,
    TerminalBench,
    Perplexity,
}

#[derive(Debug, Serialize, Clone)]
pub struct TaskResult {
    pub task_id: String,
    pub suite: Suite,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub completion: String,
    pub correct: Option<bool>,
    pub latency_ms: f64,
}

pub trait Backend {
    fn complete(&self, prompt: &str, max_tokens: usize) -> (String, f64);
}

pub fn run_suite<B: Backend>(suite: Suite, backend: &B, tasks: &[TaskSpec]) -> Vec<TaskResult> {
    tasks
        .iter()
        .filter(|t| t.suite == suite)
        .map(|t| {
            let prompt_tokens = t.prompt.split_whitespace().count();
            let (completion, latency_ms) = backend.complete(&t.prompt, 128);
            let correct = t
                .expected
                .as_ref()
                .map(|exp| completion.trim().contains(exp.trim()));
            TaskResult {
                task_id: t.id.clone(),
                suite,
                prompt_tokens,
                completion_tokens: completion.split_whitespace().count(),
                completion,
                correct,
                latency_ms,
            }
        })
        .collect()
}

pub use fixture_backend::OracleBackend;
