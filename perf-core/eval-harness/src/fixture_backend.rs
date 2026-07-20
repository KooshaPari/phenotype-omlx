//! Deterministic backend for fixture-driven evaluation without GPU metal.
//!
//! [`OracleBackend`] returns each task's expected multiple-choice letter so
//! `run_suite` can be exercised end-to-end against checked-in task loaders
//! without a live model runtime.

use crate::backend::{Backend, BackendError, Completion, Likelihood};
use crate::TaskSpec;

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
    fn complete(
        &self,
        prompt: &str,
        _max_tokens: usize,
    ) -> std::result::Result<Completion, BackendError> {
        let text = self.expected_for_prompt(prompt);
        Ok(Completion {
            text,
            prompt_tokens: 0,
            completion_tokens: 1,
            latency_ms: 0.0,
        })
    }

    fn log_likelihood(
        &self,
        prompt: &str,
        continuation: &str,
    ) -> std::result::Result<Likelihood, BackendError> {
        let expected = self.expected_for_prompt(prompt);
        let log_probability = if continuation.trim() == expected.trim() {
            0.0
        } else {
            -10.0
        };
        Ok(Likelihood {
            log_probability,
            token_count: 1,
            latency_ms: 0.0,
        })
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
            suite: Suite::Mmlu,
            prompt: "Pick one\nAnswer:".into(),
            expected: Some("B".into()),
            choices: vec!["A".into(), "B".into()],
            criteria: None,
        }];
        let backend = OracleBackend::new(&tasks);
        let completion = backend.complete("Pick one\nAnswer:", 8).unwrap();
        assert_eq!(completion.text, "B");
        assert_eq!(completion.latency_ms, 0.0);
    }
}
