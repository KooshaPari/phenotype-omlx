use eval_harness::backend::{
    BackendError, ChoiceLogprobs, Completion, Likelihood, LogprobResult, TokenLogprob,
};
use eval_harness::{run_mcq_suite, score_multiple_choice, Backend, EvalError, Suite, TaskSpec};

struct LogprobBackend {
    logprobs: Vec<f64>,
}

impl Backend for LogprobBackend {
    fn complete(&self, _: &str, _: usize) -> Result<Completion, BackendError> {
        Ok(Completion {
            text: String::new(),
            prompt_tokens: 0,
            completion_tokens: 0,
            latency_ms: 0.0,
        })
    }

    fn log_likelihood(&self, _: &str, _: &str) -> Result<Likelihood, BackendError> {
        Ok(Likelihood {
            log_probability: 0.0,
            token_count: 1,
            latency_ms: 0.0,
        })
    }

    fn complete_with_logprobs(
        &self,
        _: &str,
        choices: &[String],
    ) -> Result<LogprobResult, BackendError> {
        let choice_logprobs: Vec<ChoiceLogprobs> = choices
            .iter()
            .zip(&self.logprobs)
            .map(|(choice, &lp)| ChoiceLogprobs {
                choice: choice.clone(),
                logprob: lp,
                tokens: vec![TokenLogprob {
                    token: choice.clone(),
                    logprob: lp,
                    bytes: None,
                }],
            })
            .collect();
        Ok(LogprobResult {
            choices: choice_logprobs,
        })
    }
}

struct UnimplementedBackend;

impl Backend for UnimplementedBackend {
    fn complete(&self, _: &str, _: usize) -> Result<Completion, BackendError> {
        Ok(Completion {
            text: String::new(),
            prompt_tokens: 0,
            completion_tokens: 0,
            latency_ms: 0.0,
        })
    }

    fn log_likelihood(&self, _: &str, _: &str) -> Result<Likelihood, BackendError> {
        Ok(Likelihood {
            log_probability: 0.0,
            token_count: 1,
            latency_ms: 0.0,
        })
    }
}

fn mcq_task(id: &str, prompt: &str, choices: &[&str], expected: &str) -> TaskSpec {
    TaskSpec::multiple_choice(id, Suite::Mmlu, prompt, choices.iter().copied(), expected)
}

#[test]
fn score_returns_1_when_best_matches_correct_answer() {
    let logprobs = LogprobResult {
        choices: vec![
            ChoiceLogprobs {
                choice: "A".into(),
                logprob: -2.0,
                tokens: vec![],
            },
            ChoiceLogprobs {
                choice: "B".into(),
                logprob: -0.3,
                tokens: vec![],
            },
        ],
    };
    assert_eq!(score_multiple_choice(&logprobs, "B"), 1.0);
}

#[test]
fn score_returns_0_when_best_does_not_match() {
    let logprobs = LogprobResult {
        choices: vec![
            ChoiceLogprobs {
                choice: "A".into(),
                logprob: -0.1,
                tokens: vec![],
            },
            ChoiceLogprobs {
                choice: "B".into(),
                logprob: -3.0,
                tokens: vec![],
            },
        ],
    };
    assert_eq!(score_multiple_choice(&logprobs, "B"), 0.0);
}

#[test]
fn score_handles_empty_choices() {
    let logprobs = LogprobResult { choices: vec![] };
    assert_eq!(score_multiple_choice(&logprobs, "A"), 0.0);
}

#[test]
fn score_is_case_insensitive() {
    let logprobs = LogprobResult {
        choices: vec![ChoiceLogprobs {
            choice: "a".into(),
            logprob: -0.1,
            tokens: vec![],
        }],
    };
    assert_eq!(score_multiple_choice(&logprobs, "A"), 1.0);
}

#[test]
fn score_handles_whitespace_in_expected() {
    let logprobs = LogprobResult {
        choices: vec![ChoiceLogprobs {
            choice: "C".into(),
            logprob: -0.5,
            tokens: vec![],
        }],
    };
    assert_eq!(score_multiple_choice(&logprobs, "  C  "), 1.0);
}

#[test]
fn score_ties_keep_last_choice() {
    let logprobs = LogprobResult {
        choices: vec![
            ChoiceLogprobs {
                choice: "A".into(),
                logprob: -1.0,
                tokens: vec![],
            },
            ChoiceLogprobs {
                choice: "B".into(),
                logprob: -1.0,
                tokens: vec![],
            },
        ],
    };
    assert_eq!(score_multiple_choice(&logprobs, "B"), 1.0);
    assert_eq!(score_multiple_choice(&logprobs, "A"), 0.0);
}

#[test]
fn run_mcq_suite_selects_highest_logprob_choice() {
    let backend = LogprobBackend {
        logprobs: vec![-3.0, -0.2, -1.5],
    };
    let tasks = vec![mcq_task(
        "q1",
        "What is 2+2?\nA) 3\nB) 4\nC) 5\nAnswer:",
        &["3", "4", "5"],
        "B",
    )];
    let results = run_mcq_suite(Suite::Mmlu, &backend, &tasks).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].correct);
    assert_eq!(results[0].matched_answer.as_deref(), Some("B"));
    assert_eq!(results[0].score, 1.0);
}

#[test]
fn run_mcq_suite_scores_incorrect_answer() {
    let backend = LogprobBackend {
        logprobs: vec![-0.1, -3.0, -2.0],
    };
    let tasks = vec![mcq_task(
        "q1",
        "What is 2+2?\nA) 3\nB) 4\nC) 5\nAnswer:",
        &["3", "4", "5"],
        "B",
    )];
    let results = run_mcq_suite(Suite::Mmlu, &backend, &tasks).unwrap();
    assert_eq!(results.len(), 1);
    assert!(!results[0].correct);
    assert_eq!(results[0].matched_answer.as_deref(), Some("A"));
    assert_eq!(results[0].score, 0.0);
}

#[test]
fn run_mcq_suite_pass_at_one_aggregate() {
    let backend_correct = LogprobBackend {
        logprobs: vec![-3.0, -0.2],
    };
    let backend_wrong = LogprobBackend {
        logprobs: vec![-0.1, -3.0],
    };
    let tasks = vec![
        mcq_task("q1", "Q1\nA) x\nB) y", &["x", "y"], "B"),
        mcq_task("q2", "Q2\nA) a\nB) b", &["a", "b"], "B"),
    ];

    let results_correct = run_mcq_suite(Suite::Mmlu, &backend_correct, &tasks).unwrap();
    let pass_at_one_correct =
        results_correct.iter().filter(|r| r.correct).count() as f64 / results_correct.len() as f64;
    assert_eq!(pass_at_one_correct, 1.0);

    let results_wrong = run_mcq_suite(Suite::Mmlu, &backend_wrong, &tasks).unwrap();
    let pass_at_one_wrong =
        results_wrong.iter().filter(|r| r.correct).count() as f64 / results_wrong.len() as f64;
    assert_eq!(pass_at_one_wrong, 0.0);
}

#[test]
fn run_mcq_suite_filters_non_mcq_tasks() {
    let backend = LogprobBackend {
        logprobs: vec![-0.1],
    };
    let tasks = vec![TaskSpec {
        id: "open".into(),
        suite: Suite::Mmlu,
        prompt: "Write a poem".into(),
        expected: Some("roses are red".into()),
        choices: vec![],
        criteria: None,
    }];
    let results = run_mcq_suite(Suite::Mmlu, &backend, &tasks).unwrap();
    assert!(results.is_empty());
}

#[test]
fn run_mcq_suite_empty_choices_returns_error() {
    struct EmptyLogprobBackend;
    impl Backend for EmptyLogprobBackend {
        fn complete(&self, _: &str, _: usize) -> Result<Completion, BackendError> {
            Ok(Completion {
                text: String::new(),
                prompt_tokens: 0,
                completion_tokens: 0,
                latency_ms: 0.0,
            })
        }
        fn log_likelihood(&self, _: &str, _: &str) -> Result<Likelihood, BackendError> {
            Ok(Likelihood {
                log_probability: 0.0,
                token_count: 1,
                latency_ms: 0.0,
            })
        }
        fn complete_with_logprobs(
            &self,
            _: &str,
            _: &[String],
        ) -> Result<LogprobResult, BackendError> {
            Ok(LogprobResult { choices: vec![] })
        }
    }
    let tasks = vec![mcq_task("q1", "Q?", &["a", "b"], "A")];
    let error = run_mcq_suite(Suite::Mmlu, &EmptyLogprobBackend, &tasks).unwrap_err();
    assert!(matches!(
        error,
        EvalError::Backend {
            source: BackendError::InvalidResponse { .. },
            ..
        }
    ));
}

#[test]
fn default_complete_with_logprobs_returns_unavailable() {
    let error = UnimplementedBackend
        .complete_with_logprobs("prompt", &["A".into()])
        .unwrap_err();
    assert!(matches!(error, BackendError::Unavailable { .. }));
}

#[test]
fn token_logprob_preserves_bytes() {
    let logprobs = LogprobResult {
        choices: vec![ChoiceLogprobs {
            choice: "hello".into(),
            logprob: -0.5,
            tokens: vec![TokenLogprob {
                token: "hello".into(),
                logprob: -0.5,
                bytes: Some(b"hello".to_vec()),
            }],
        }],
    };
    let token = &logprobs.choices[0].tokens[0];
    assert_eq!(token.bytes.as_deref(), Some(b"hello".as_slice()));
}
