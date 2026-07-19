//! Deterministic runners for completion and multiple-choice evaluations.

use crate::{evaluate, Backend, EvalError, Result, Suite, TaskResult, TaskSpec};

/// Run completion-based tasks for one suite.
pub fn run_suite<B: Backend>(suite: Suite, backend: &B, tasks: &[TaskSpec]) -> Result<Vec<TaskResult>> {
    tasks
        .iter()
        .filter(|task| task.suite == suite)
        .map(|task| {
            let completion = backend.complete(&task.prompt, 128)?;
            let mut result = evaluate(task, &completion.text)?;
            result.latency_ms = completion.latency_ms;
            Ok(result)
        })
        .collect()
}

/// Run multiple-choice tasks by selecting the choice with the greatest log likelihood.
///
/// Ties retain the first choice, so results are stable and preserve dataset order.
pub fn run_multiple_choice_suite<B: Backend>(
    suite: Suite,
    backend: &B,
    tasks: &[TaskSpec],
) -> Result<Vec<TaskResult>> {
    tasks
        .iter()
        .filter(|task| task.suite == suite)
        .map(|task| run_multiple_choice_task(backend, task))
        .collect()
}

fn run_multiple_choice_task<B: Backend>(backend: &B, task: &TaskSpec) -> Result<TaskResult> {
    if !task.is_multiple_choice() {
        return Err(EvalError::invalid_task(&task.id, "multiple-choice task has no choices"));
    }

    let mut best = (0_usize, f64::NEG_INFINITY);
    for (index, choice) in task.choices.iter().enumerate() {
        let likelihood = backend.log_likelihood(&task.prompt, choice)?;
        if likelihood > best.1 {
            best = (index, likelihood);
        }
    }

    let answer = char::from(b'A' + best.0 as u8).to_string();
    evaluate(task, &answer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackendCompletion, EvalError};

    struct FailingBackend;

    impl Backend for FailingBackend {
        fn complete(&self, _: &str, _: usize) -> Result<BackendCompletion> {
            Err(EvalError::backend("unavailable"))
        }

        fn log_likelihood(&self, _: &str, _: &str) -> Result<f64> {
            Err(EvalError::backend("unavailable"))
        }
    }

    #[test]
    fn completion_runner_propagates_backend_errors() {
        let task = TaskSpec::multiple_choice("q", Suite::Mmlu, "Question", ["one"], "A");
        assert!(matches!(run_suite(Suite::Mmlu, &FailingBackend, &[task]), Err(EvalError::Backend { .. })));
    }
}
