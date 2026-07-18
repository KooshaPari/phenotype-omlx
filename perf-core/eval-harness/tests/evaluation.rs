use eval_harness::{
    evaluate, normalize_answer, score_choice, score_exact, EvaluationReport, Suite, TaskResult,
    TaskSpec,
};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

#[test]
fn mmlu_csv_loader_builds_stable_multiple_choice_tasks() {
    let tasks = eval_harness::mmlu::load_csv(fixture("mmlu.csv")).unwrap();
    assert_eq!(tasks.iter().map(|task| task.id.as_str()).collect::<Vec<_>>(), [
        "mmlu_anatomy_1",
        "mmlu_physics_2",
    ]);
    assert_eq!(tasks[0].suite, Suite::MMLU);
    assert_eq!(tasks[0].choices, vec!["Cranial", "Thoracic", "Abdominal", "Pelvic"]);
    assert_eq!(tasks[0].expected.as_deref(), Some("B"));
}

#[test]
fn gpqa_jsonl_loader_is_sorted_and_preserves_choices() {
    let tasks = eval_harness::gpqa::load_jsonl(fixture("gpqa.jsonl")).unwrap();
    assert_eq!(tasks.iter().map(|task| task.id.as_str()).collect::<Vec<_>>(), [
        "gpqa_biology-1",
        "gpqa_chemistry-1",
    ]);
    assert_eq!(tasks[1].choices[0], "Fluorine");
    assert_eq!(tasks[1].suite, Suite::GPQA);
}

#[test]
fn yaml_loader_reads_terminal_bench_criteria_without_execution() {
    let tasks = eval_harness::terminal_bench::load_yaml(fixture("terminal-bench/task.yaml")).unwrap();
    let criteria = tasks[0].criteria.as_ref().unwrap();
    assert_eq!(criteria.expected_commands, vec!["grep -r 'FIXME' ."]);
    assert_eq!(tasks[0].suite, Suite::TerminalBench);

    let result = evaluate(&tasks[0], "grep -r 'FIXME' .\nsrc/main.rs:FIXME").unwrap();
    assert!(result.correct);
    assert_eq!(result.score, 1.0);
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
fn report_is_deterministic_and_serde_round_trips() {
    let tasks = vec![
        TaskSpec::multiple_choice("b", Suite::MMLU, "Q2", vec!["x"], "A"),
        TaskSpec::multiple_choice("a", Suite::MMLU, "Q1", vec!["x"], "A"),
    ];
    let results = tasks.iter().map(|task| evaluate(task, "A").unwrap()).collect::<Vec<_>>();
    let report = EvaluationReport::from_results(Suite::MMLU, results);
    assert_eq!(report.results.iter().map(|r| r.task_id.as_str()).collect::<Vec<_>>(), ["a", "b"]);
    assert_eq!(report.accuracy, 1.0);

    let encoded = serde_json::to_string(&report).unwrap();
    let decoded: EvaluationReport = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, report);
}

#[test]
fn task_result_serializes_all_evaluation_fields() {
    let result = TaskResult {
        task_id: "task".into(),
        suite: Suite::GPQA,
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
}
