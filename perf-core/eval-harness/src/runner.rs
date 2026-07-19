//! Deterministic runners for completion and multiple-choice evaluations.

use crate::{
    backend::{BackendError, Likelihood},
    evaluate, Backend, EvalError, Result, Suite, TaskResult, TaskSpec,
};

/// Run all tasks for one suite using completion or likelihood scoring as appropriate.
pub fn run_suite<B: Backend>(
    suite: Suite,
    backend: &B,
    tasks: &[TaskSpec],
) -> Result<Vec<TaskResult>> {
    tasks
        .iter()
        .filter(|task| task.suite == suite)
        .map(|task| run_task(backend, task))
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

fn run_task<B: Backend>(backend: &B, task: &TaskSpec) -> Result<TaskResult> {
    if task.is_multiple_choice() {
        return run_multiple_choice_task(backend, task);
    }

    let completion = backend
        .complete(&task.prompt, 128)
        .map_err(|source| backend_error(task, source))?;
    let mut result = evaluate(task, &completion.text)?;
    result.prompt_tokens = completion.prompt_tokens;
    result.completion_tokens = completion.completion_tokens;
    result.latency_ms = completion.latency_ms;
    Ok(result)
}

fn run_multiple_choice_task<B: Backend>(backend: &B, task: &TaskSpec) -> Result<TaskResult> {
    let mut best: Option<(usize, Likelihood)> = None;
    let mut latency_ms = 0.0;

    for index in 0..task.choices.len() {
        let label = char::from(b'A' + index as u8);
        let likelihood = backend
            .log_likelihood(&task.prompt, &format!(" {label}"))
            .map_err(|source| backend_error(task, source))?;
        if !likelihood.log_probability.is_finite() || !likelihood.latency_ms.is_finite() {
            return Err(backend_error(
                task,
                BackendError::InvalidResponse {
                    message: format!("non-finite likelihood for choice {label}"),
                },
            ));
        }
        latency_ms += likelihood.latency_ms;
        if best
            .as_ref()
            .is_none_or(|(_, current)| likelihood.log_probability > current.log_probability)
        {
            best = Some((index, likelihood));
        }
    }

    let (index, likelihood) = best.ok_or_else(|| {
        backend_error(
            task,
            BackendError::InvalidResponse {
                message: "multiple-choice task has no choices".into(),
            },
        )
    })?;
    let label = char::from(b'A' + index as u8).to_string();
    let mut result = evaluate(task, &label)?;
    result.correct = task.expected.as_deref() == Some(label.as_str());
    result.score = f64::from(result.correct);
    result.prompt_tokens = task.prompt.split_whitespace().count().saturating_sub(1);
    result.completion_tokens = likelihood.token_count;
    result.latency_ms = latency_ms;
    result.matched_answer = Some(label);
    Ok(result)
}

fn backend_error(task: &TaskSpec, source: BackendError) -> EvalError {
    EvalError::backend(&task.id, source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::BackendError, BackendCompletion, EvalError};

    struct FailingBackend;

    impl Backend for FailingBackend {
        fn complete(
            &self,
            _: &str,
            _: usize,
        ) -> std::result::Result<BackendCompletion, BackendError> {
            Err(BackendError::Unavailable {
                message: "unavailable".into(),
            })
        }

        fn log_likelihood(
            &self,
            _: &str,
            _: &str,
        ) -> std::result::Result<Likelihood, BackendError> {
            Err(BackendError::Unavailable {
                message: "unavailable".into(),
            })
        }
    }

    #[test]
    fn completion_runner_propagates_backend_errors() {
        let task = TaskSpec::multiple_choice("q", Suite::Mmlu, "Question", ["one"], "A");
        assert!(matches!(
            run_suite(Suite::Mmlu, &FailingBackend, &[task]),
            Err(EvalError::Backend { .. })
        ));
    }
}
