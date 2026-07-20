//! End-to-end runner tests wiring [`eval_harness::Backend`] to fixture loaders.

use eval_harness::{OracleBackend, Suite, gpqa, mmlu, run_suite};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn run_suite_scores_mmlu_fixtures_via_oracle_backend() {
    let dataset = mmlu::load_csv(fixture("mmlu.csv")).unwrap();
    let tasks = dataset.as_tasks();
    let backend = OracleBackend::new(tasks);
    let results = run_suite(Suite::Mmlu, &backend, tasks).unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|result| result.correct));
}

#[test]
fn run_suite_scores_gpqa_fixtures_via_oracle_backend() {
    let dataset = gpqa::load_jsonl(fixture("gpqa.jsonl")).unwrap();
    let tasks = dataset.as_tasks();
    let backend = OracleBackend::new(tasks);
    let results = run_suite(Suite::Gpqa, &backend, tasks).unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().all(|result| result.correct));
}
