//! Deterministic backend for fixture-driven evaluation without GPU metal.
//!
//! [`OracleBackend`] returns each task's expected multiple-choice letter so
//! `run_suite` can be exercised end-to-end against checked-in task loaders
//! without a live model runtime.

use crate::{Backend, TaskSpec};

/// Backend that completes with the expected answer for a matching prompt.
///
/// Lookup is by exact `prompt` match against the task list supplied at
/// construction time. When no task matches, completion is empty.
pub struct OracleBackend<'a> {
    tasks: &'a [TaskSpec],
}

impl<'a> OracleBackend<'a> {
    pub fn new(tasks: &'a [TaskSpec]) -> Self {
        Self { tasks }
    }

    fn expected_for_prompt(&self, prompt: &str) -> String {
        self.tasks
            .iter()
            .find(|task| task.prompt == prompt)
            .and_then(|task| task.expected.clone())
            .unwrap_or_default()
    }
}

impl Backend for OracleBackend<'_> {
    fn complete(&self, prompt: &str, _max_tokens: usize) -> (String, f64) {
        (self.expected_for_prompt(prompt), 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Suite;

    #[test]
    fn oracle_backend_returns_expected_letter() {
        let tasks = [TaskSpec {
            id: "q1".into(),
            suite: Suite::MMLU,
            prompt: "Pick one\nAnswer:".into(),
            expected: Some("B".into()),
            choices: None,
        }];
        let backend = OracleBackend::new(&tasks);
        let (completion, latency_ms) = backend.complete("Pick one\nAnswer:", 8);
        assert_eq!(completion, "B");
        assert_eq!(latency_ms, 0.0);
    }
}
