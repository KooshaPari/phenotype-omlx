//! Integration tests for the eval-harness public API.
//!
//! These tests cover:
//! - Loaders produce deterministic, provenance-tagged [`eval_harness::Dataset`]s
//!   for the MMLU, GPQA, and terminal-bench fixtures checked into the repo.
//! - Normalization, exact scoring, and choice scoring are stable across the
//!   edge cases the harness contract requires.
//! - [`eval_harness::EvaluationReport`] is deterministic, serde-stable, and
//!   carries the per-task metrics downstream tooling relies on.

use eval_harness::{
    evaluate, normalize_answer, score_choice, score_exact, EvaluationReport, Suite, TaskResult,
    TaskSpec,
};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Source revision used for fixture-driven tests. Real callers would record
/// the upstream dataset commit/tag here.
const FIXTURE_REV: &str = "fixture-v1";
const FIXTURE_SPLIT: &str = "test";

#[test]
fn mmlu_csv_loader_builds_stable_multiple_choice_tasks() {
    let dataset = eval_harness::mmlu::load_csv_with_provenance(
        fixture("mmlu.csv"),
        FIXTURE_REV,
        FIXTURE_SPLIT,
    )
    .expect("fixture must load");
    assert_eq!(
        dataset
            .iter()
            .map(|task| task.id.as_str())
            .collect::<Vec<_>>(),
        ["mmlu_anatomy_1", "mmlu_physics_2"]
    );
    assert_eq!(dataset[0].suite, Suite::Mmlu);
    assert_eq!(
        dataset[0].choices,
        vec!["Cranial", "Thoracic", "Abdominal", "Pelvic"]
    );
    assert_eq!(dataset[0].expected.as_deref(), Some("B"));
    assert!(dataset[0].is_multiple_choice());
    // No criteria should be set for multiple-choice MMLU rows.
    assert!(dataset[0].criteria.is_none());
    // Dataset provenance is populated from the file.
    let prov = dataset.provenance();
    assert_eq!(prov.source_revision, FIXTURE_REV);
    assert_eq!(prov.split, FIXTURE_SPLIT);
    assert_eq!(prov.task_count, 2);
    assert_eq!(prov.sha256.len(), 64);
}

#[test]
fn gpqa_jsonl_loader_is_sorted_and_preserves_choices() {
    let dataset = eval_harness::gpqa::load_jsonl_with_provenance(
        fixture("gpqa.jsonl"),
        FIXTURE_REV,
        FIXTURE_SPLIT,
    )
    .expect("fixture must load");
    assert_eq!(
        dataset
            .iter()
            .map(|task| task.id.as_str())
            .collect::<Vec<_>>(),
        ["gpqa_biology-1", "gpqa_chemistry-1"]
    );
    assert_eq!(dataset[1].choices[0], "Fluorine");
    assert_eq!(dataset[1].suite, Suite::Gpqa);
    assert!(dataset[1].criteria.is_none());
    assert_eq!(dataset.provenance().split, FIXTURE_SPLIT);
}

#[test]
fn yaml_loader_reads_terminal_bench_criteria_without_execution() {
    let dataset = eval_harness::terminal_bench::load_yaml_with_provenance(
        fixture("terminal-bench/task.yaml"),
        FIXTURE_REV,
        FIXTURE_SPLIT,
    )
    .expect("fixture must load");
    let criteria = dataset[0].criteria.as_ref().expect("criteria is required");
    assert_eq!(criteria.expected_commands, vec!["grep -r 'FIXME' ."]);
    assert_eq!(dataset[0].suite, Suite::TerminalBench);

    // evaluate() must be pure: no command execution, only substring gating.
    let result = evaluate(&dataset[0], "grep -r 'FIXME' .\nsrc/main.rs:FIXME").unwrap();
    assert!(result.correct);
    assert_eq!(result.score, 1.0);

    // Forbidden output should fail the task even if required output is present.
    let bad = evaluate(
        &dataset[0],
        "grep -r 'FIXME' .\nsrc/main.rs:FIXME\npermission denied on /etc",
    )
    .unwrap();
    assert!(!bad.correct);
}

#[test]
fn normalization_and_scoring_are_exact_and_choice_aware() {
    assert_eq!(normalize_answer("  The Answer!\n"), "the answer");
    assert!(score_exact(" Newton. ", "newton"));
    assert!(score_choice("The correct answer is (b).", "B", 4));
    assert!(!score_choice("The correct answer is C.", "B", 4));
    assert!(!score_exact("not exact", "exact"));
}

#[test]
fn normalize_handles_empty_and_whitespace_only() {
    assert_eq!(normalize_answer(""), "");
    assert_eq!(normalize_answer("   "), "");
    assert_eq!(normalize_answer("\t\r\n"), "");
}

#[test]
fn normalize_is_idempotent() {
    let once = normalize_answer("  Hello, WORLD!!! ");
    let twice = normalize_answer(&once);
    assert_eq!(once, twice);
}

#[test]
fn exact_score_rejects_partial_match() {
    // Exact scoring is a normalized full-string equality, not a substring test.
    assert!(!score_exact("The answer is Newton", "Newton"));
}

#[test]
fn choice_score_prefers_parenthesized_marker() {
    assert!(score_choice("(b)", "B", 4));
    assert!(score_choice("Reasoning... (b) end.", "B", 4));
}

#[test]
fn choice_score_falls_back_to_trailing_letter() {
    assert!(score_choice("Reasoning then I think it's A.", "A", 4));
    assert!(!score_choice("Reasoning then I think it's A.", "B", 4));
}

#[test]
fn choice_score_rejects_letter_outside_range() {
    // Only A and B are valid with num_choices=2.
    assert!(!score_choice("The answer is C.", "C", 2));
    assert!(!score_choice("(Z)", "Z", 4));
}

#[test]
fn choice_score_rejects_zero_or_oversized_choice_count() {
    assert!(!score_choice("(A)", "A", 0));
    assert!(!score_choice("(A)", "A", 27));
}

#[test]
fn choice_score_rejects_non_letter_expected() {
    assert!(!score_choice("(A)", "1", 4));
    assert!(!score_choice("(A)", "", 4));
}

#[test]
fn report_is_deterministic_and_serde_round_trips() {
    let tasks = [
        TaskSpec::multiple_choice("b", Suite::Mmlu, "Q2", vec!["x"], "A"),
        TaskSpec::multiple_choice("a", Suite::Mmlu, "Q1", vec!["x"], "A"),
    ];
    let results: Vec<TaskResult> = tasks
        .iter()
        .map(|task| evaluate(task, "A").unwrap())
        .collect();
    let report = EvaluationReport::from_results(Suite::Mmlu, results);
    assert_eq!(
        report
            .results
            .iter()
            .map(|r| r.task_id.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_eq!(report.task_count, 2);
    assert_eq!(report.correct_count, 2);
    assert_eq!(report.accuracy, 1.0);
    assert_eq!(report.mean_score, 1.0);

    let encoded = serde_json::to_string(&report).unwrap();
    let decoded: EvaluationReport = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, report);
}

#[test]
fn report_handles_empty_input() {
    let report = EvaluationReport::from_results(Suite::Gpqa, vec![]);
    assert_eq!(report.task_count, 0);
    assert_eq!(report.correct_count, 0);
    assert_eq!(report.accuracy, 0.0);
    assert_eq!(report.mean_score, 0.0);
    assert!(report.results.is_empty());
    let encoded = serde_json::to_string(&report).unwrap();
    let decoded: EvaluationReport = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, report);
}

#[test]
fn report_handles_partial_correctness() {
    let mut r1 = evaluate(
        &TaskSpec::multiple_choice("a", Suite::Mmlu, "Q", vec!["x"], "A"),
        "A",
    )
    .unwrap();
    r1.score = 1.0;
    let mut r2 = evaluate(
        &TaskSpec::multiple_choice("b", Suite::Mmlu, "Q", vec!["x"], "B"),
        "A",
    )
    .unwrap();
    r2.score = 0.5;
    let report = EvaluationReport::from_results(Suite::Mmlu, vec![r1, r2]);
    assert_eq!(report.task_count, 2);
    assert_eq!(report.correct_count, 1);
    assert_eq!(report.accuracy, 0.5);
    assert_eq!(report.mean_score, 0.75);
}

#[test]
fn task_result_serializes_all_evaluation_fields() {
    let result = TaskResult {
        task_id: "task".into(),
        suite: Suite::Gpqa,
        prompt_tokens: 2,
        completion_tokens: 1,
        completion: "B".into(),
        normalized_completion: "b".into(),
        correct: true,
        score: 1.0,
        latency_ms: 2.5,
        matched_answer: Some("B".into()),
    };
    let value = serde_json::to_value(result).unwrap();
    assert_eq!(value["matched_answer"], "B");
    assert_eq!(value["score"], 1.0);
    assert_eq!(value["correct"], true);
    assert_eq!(value["latency_ms"], 2.5);
    assert_eq!(value["prompt_tokens"], 2);
    assert_eq!(value["completion_tokens"], 1);
    assert_eq!(value["normalized_completion"], "b");
}

#[test]
fn evaluate_without_expected_or_criteria_is_zero() {
    let task = TaskSpec {
        id: "x".into(),
        suite: Suite::Perplexity,
        prompt: "p".into(),
        expected: None,
        choices: vec![],
        criteria: None,
    };
    let r = evaluate(&task, "anything").unwrap();
    assert!(!r.correct);
    assert_eq!(r.score, 0.0);
    assert_eq!(r.matched_answer, None);
}

#[test]
fn evaluate_exact_does_not_use_choice_letter_rules() {
    // "(b)" should NOT match an exact-scored task that expects "B"; exact
    // scoring compares the normalized completion string, not choice markers.
    let task = TaskSpec {
        id: "t".into(),
        suite: Suite::Gpqa,
        prompt: "p".into(),
        expected: Some("B".into()),
        choices: vec![],
        criteria: None,
    };
    let r = evaluate(&task, "(b)").unwrap();
    assert!(!r.correct);
}
