use eval_harness::{
    backend::{BackendError, Completion, Likelihood},
    run_suite, Backend, EvalError, Suite, TaskSpec,
};
use std::cell::RefCell;

#[derive(Default)]
struct RecordingBackend {
    likelihoods: Vec<f64>,
    continuations: RefCell<Vec<String>>,
    completion: Option<Result<Completion, BackendError>>,
}

impl Backend for RecordingBackend {
    fn complete(&self, _prompt: &str, _max_tokens: usize) -> Result<Completion, BackendError> {
        self.completion.clone().unwrap_or_else(|| {
            Ok(Completion {
                text: "ok".into(),
                prompt_tokens: 1,
                completion_tokens: 1,
                latency_ms: 2.0,
            })
        })
    }

    fn log_likelihood(
        &self,
        _prompt: &str,
        continuation: &str,
    ) -> Result<Likelihood, BackendError> {
        self.continuations.borrow_mut().push(continuation.into());
        let index = self.continuations.borrow().len() - 1;
        Ok(Likelihood {
            log_probability: self.likelihoods[index],
            token_count: 1,
            latency_ms: 1.5,
        })
    }
}

fn choice_task(suite: Suite) -> TaskSpec {
    TaskSpec::multiple_choice(
        "q1",
        suite,
        "Question\nA) x\nB) y\nAnswer:",
        ["x", "y"],
        "B",
    )
}

#[test]
fn mmlu_uses_answer_label_log_likelihoods() {
    let backend = RecordingBackend {
        likelihoods: vec![-4.0, -0.2],
        ..Default::default()
    };

    let results = run_suite(Suite::Mmlu, &backend, &[choice_task(Suite::Mmlu)]).unwrap();

    assert_eq!(&*backend.continuations.borrow(), &[" A", " B"]);
    assert_eq!(results[0].completion, "B");
    assert_eq!(results[0].matched_answer.as_deref(), Some("B"));
    assert!(results[0].correct);
    assert_eq!(results[0].latency_ms, 3.0);
}

#[test]
fn gpqa_scores_the_highest_log_likelihood_not_generated_text() {
    let backend = RecordingBackend {
        likelihoods: vec![-0.1, -3.0],
        completion: Some(Ok(Completion {
            text: "B".into(),
            prompt_tokens: 99,
            completion_tokens: 99,
            latency_ms: 99.0,
        })),
        ..Default::default()
    };

    let results = run_suite(Suite::Gpqa, &backend, &[choice_task(Suite::Gpqa)]).unwrap();

    assert_eq!(results[0].completion, "A");
    assert!(!results[0].correct);
    assert_eq!(results[0].prompt_tokens, 5);
    assert_eq!(results[0].completion_tokens, 1);
}

#[test]
fn backend_errors_retain_task_context_and_typed_source() {
    let backend = RecordingBackend {
        completion: Some(Err(BackendError::Unavailable {
            message: "model is loading".into(),
        })),
        ..Default::default()
    };
    let task = TaskSpec {
        id: "terminal-7".into(),
        suite: Suite::TerminalBench,
        prompt: "do work".into(),
        expected: Some("ok".into()),
        choices: vec![],
        criteria: None,
    };

    let error = run_suite(Suite::TerminalBench, &backend, &[task]).unwrap_err();

    match error {
        EvalError::Backend {
            task_id,
            source: BackendError::Unavailable { message },
        } => {
            assert_eq!(task_id, "terminal-7");
            assert_eq!(message, "model is loading");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn non_finite_likelihood_is_rejected_instead_of_silently_scored() {
    let backend = RecordingBackend {
        likelihoods: vec![f64::NAN, -1.0],
        ..Default::default()
    };

    let error = run_suite(Suite::Mmlu, &backend, &[choice_task(Suite::Mmlu)]).unwrap_err();

    assert!(matches!(
        error,
        EvalError::Backend {
            source: BackendError::InvalidResponse { .. },
            ..
        }
    ));
}

#[test]
fn multiple_choice_scoring_uses_direct_label_equality() {
    let backend = RecordingBackend {
        likelihoods: vec![-0.1, -3.0],
        ..Default::default()
    };
    let mut task = choice_task(Suite::Mmlu);
    task.expected = Some("the answer is A".into());

    let results = run_suite(Suite::Mmlu, &backend, &[task]).unwrap();

    assert_eq!(results[0].completion, "A");
    assert!(!results[0].correct);
    assert_eq!(results[0].matched_answer.as_deref(), Some("A"));
}
