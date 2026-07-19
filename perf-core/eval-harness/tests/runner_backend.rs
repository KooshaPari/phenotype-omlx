//! End-to-end runner tests wiring [`eval_harness::Backend`] to fixture loaders.

use eval_harness::{OracleBackend, Suite, gpqa, mmlu, run_suite};

#[test]
fn run_suite_scores_mmlu_fixtures_via_oracle_backend() {
    let tasks = mmlu::load_tasks();
    let backend = OracleBackend::new(&tasks);
    let results = run_suite(Suite::MMLU, &backend, &tasks);
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|result| result.correct == Some(true)));
}

#[test]
fn run_suite_scores_gpqa_fixtures_via_oracle_backend() {
    let tasks = gpqa::load_tasks();
    let backend = OracleBackend::new(&tasks);
    let results = run_suite(Suite::GPQA, &backend, &tasks);
    assert!(!results.is_empty());
    assert!(results.iter().all(|result| result.correct == Some(true)));
}
