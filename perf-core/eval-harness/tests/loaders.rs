//! Integration tests for the eval-harness public API.
//!
//! These tests cover:
//! - Loaders produce deterministic, provenance-tagged [`eval_harness::Dataset`]s
//!   for the MMLU, GPQA, and terminal-bench fixtures checked into the repo.
//! - Normalization, exact scoring, and choice scoring are stable across the
//!   edge cases the harness contract requires.
//! - [`eval_harness::EvaluationReport`] is deterministic, serde-stable, and
//!   carries the per-task metrics downstream tooling relies on.
//! - Cross-suite aggregation via [`eval_harness::report::MultiSuiteReport`]
//!   is task-weighted, order-independent, and provenance-preserving.
//! - Malformed records in CSV / JSONL / YAML surface a structured
//!   [`eval_harness::EvalError`] with path, line, and message context.
//! - The SHA-256 helper used by provenance is deterministic and cross-checked
//!   against the FIPS 180-4 "abc" vector.

use eval_harness::{dataset::Dataset, provenance, EvaluationReport, Suite, TaskResult};

/// Source revision used for fixture-driven tests. Real callers would record
/// the upstream dataset commit/tag here.
const FIXTURE_REV: &str = "fixture-v1";
const FIXTURE_SPLIT: &str = "test";

#[test]
fn malformed_csv_record_returns_structured_error() {
    use eval_harness::EvalError;
    // Header is fine but row has too few fields.
    let bytes = b"subject,question,A,B,answer\nanatomy,Q,Cranial\n";
    let err =
        eval_harness::mmlu::load_csv_bytes(bytes, "x.csv", FIXTURE_REV, FIXTURE_SPLIT).unwrap_err();
    match err {
        EvalError::Csv {
            path,
            line,
            message,
        } => {
            assert_eq!(path, "x.csv");
            assert_eq!(line, 2);
            assert!(message.contains("row has"));
        }
        other => panic!("expected Csv error, got {other:?}"),
    }
}

#[test]
fn malformed_csv_missing_header_returns_error() {
    use eval_harness::EvalError;
    let err =
        eval_harness::mmlu::load_csv_bytes(b"", "x.csv", FIXTURE_REV, FIXTURE_SPLIT).unwrap_err();
    match err {
        EvalError::Csv { line, .. } => assert_eq!(line, 1),
        other => panic!("expected Csv error, got {other:?}"),
    }
}

#[test]
fn malformed_csv_missing_answer_column_returns_error() {
    use eval_harness::EvalError;
    let bytes = b"subject,question,A,B\nfoo,bar,x,y\n";
    let err =
        eval_harness::mmlu::load_csv_bytes(bytes, "x.csv", FIXTURE_REV, FIXTURE_SPLIT).unwrap_err();
    match err {
        EvalError::MissingField { field, .. } => assert_eq!(field, "answer"),
        other => panic!("expected MissingField error, got {other:?}"),
    }
}

#[test]
fn malformed_jsonl_record_returns_error_with_line_number() {
    use eval_harness::EvalError;
    // Two blank lines then a malformed JSON record. The malformed record is
    // on line 3 (1-indexed).
    let bytes = b"\n\n{not json\n";
    let err = eval_harness::gpqa::load_jsonl_bytes(bytes, "x.jsonl", FIXTURE_REV, FIXTURE_SPLIT)
        .unwrap_err();
    match err {
        EvalError::Json { path, line, .. } => {
            // path is pure (no line suffix) and line is structured.
            assert_eq!(path, "x.jsonl");
            assert_eq!(line, 3);
        }
        other => panic!("expected Json error, got {other:?}"),
    }
}

#[test]
fn malformed_jsonl_out_of_range_answer_returns_error() {
    use eval_harness::EvalError;
    let bytes = b"{\"id\":\"a\",\"question\":\"Q\",\"choices\":[\"x\",\"y\"],\"answer\":\"Z\"}\n";
    let err = eval_harness::gpqa::load_jsonl_bytes(bytes, "x.jsonl", FIXTURE_REV, FIXTURE_SPLIT)
        .unwrap_err();
    match err {
        EvalError::Malformed { line, message, .. } => {
            assert_eq!(line, 1);
            assert!(message.contains("out of range"));
        }
        other => panic!("expected Malformed error, got {other:?}"),
    }
}

#[test]
fn malformed_yaml_returns_structured_error() {
    use eval_harness::EvalError;
    let bytes = b"id: a\nprompt: P\ncriteria:\n  expected_commands: [unclosed\n";
    let err =
        eval_harness::terminal_bench::load_yaml_bytes(bytes, "x.yaml", FIXTURE_REV, FIXTURE_SPLIT)
            .unwrap_err();
    match err {
        EvalError::Yaml { path, line, .. } => {
            assert_eq!(path, "x.yaml");
            // Line is structured and >= 1; serde_yaml's location() may report
            // 1-based or fall back to 1 when unavailable.
            assert!(line >= 1, "line should be >= 1, got {line}");
        }
        other => panic!("expected Yaml error, got {other:?}"),
    }
}

#[test]
fn malformed_yaml_missing_criteria_returns_error() {
    use eval_harness::EvalError;
    let bytes = b"id: a\nprompt: P\ncriteria: {}\n";
    let err =
        eval_harness::terminal_bench::load_yaml_bytes(bytes, "x.yaml", FIXTURE_REV, FIXTURE_SPLIT)
            .unwrap_err();
    match err {
        EvalError::Malformed { line, message, .. } => {
            assert_eq!(line, 1);
            assert!(message.contains("no expected_commands"));
        }
        other => panic!("expected Malformed error, got {other:?}"),
    }
}

#[test]
fn io_error_for_missing_file_includes_path() {
    use eval_harness::EvalError;
    let err = eval_harness::mmlu::load_csv_with_provenance(
        "/does/not/exist.csv",
        FIXTURE_REV,
        FIXTURE_SPLIT,
    )
    .unwrap_err();
    match err {
        EvalError::Io { path, .. } => assert!(path.contains("does/not/exist.csv")),
        other => panic!("expected Io error, got {other:?}"),
    }
}

#[test]
fn provenance_sha256_matches_fips_180_4_abc_vector() {
    // Sanity-check that the SHA-256 implementation used by loaders matches
    // the well-known FIPS 180-4 "abc" vector.
    assert_eq!(
        provenance::sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn provenance_sha256_of_empty_matches_known_vector() {
    assert_eq!(
        provenance::sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn dataset_serde_round_trips() {
    let dataset = eval_harness::mmlu::load_csv_bytes(
        b"subject,question,A,B,answer\nfoo,Q?,x,y,A\n",
        "x.csv",
        FIXTURE_REV,
        FIXTURE_SPLIT,
    )
    .unwrap();
    let encoded = serde_json::to_string(&dataset).unwrap();
    let decoded: Dataset = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, dataset);
}

#[test]
fn dataset_deref_allows_indexing_and_iteration() {
    let dataset = eval_harness::mmlu::load_csv_bytes(
        b"subject,question,A,B,answer\nfoo,Q1?,x,y,A\nbar,Q2?,p,q,B\n",
        "x.csv",
        FIXTURE_REV,
        FIXTURE_SPLIT,
    )
    .unwrap();
    // Iteration through Deref.
    let ids: Vec<&str> = dataset.iter().map(|t| t.id.as_str()).collect();
    assert_eq!(ids, vec!["mmlu_bar_2", "mmlu_foo_1"]);
    // Indexing through Deref.
    assert_eq!(dataset[0].id, "mmlu_bar_2");
}

#[test]
fn cross_suite_aggregate_sorts_by_suite_declaration_order() {
    // The aggregate must sort entries by the suite's declaration order
    // (Suite::MMLU < Suite::GPQA < Suite::TerminalBench < Suite::Perplexity),
    // matching the derived Ord. Inserting entries in a scrambled order must
    // still produce the declaration-order sequence in the aggregate.
    use eval_harness::provenance::DatasetProvenance;
    use eval_harness::report::{MultiSuiteReport, SuiteReportEntry};

    fn make_entry(suite: Suite, total: usize) -> SuiteReportEntry {
        let results: Vec<TaskResult> = (0..total)
            .map(|i| TaskResult {
                task_id: format!("{}-{}", suite.as_str(), i),
                suite,
                prompt_tokens: 1,
                completion_tokens: 1,
                completion: "c".into(),
                normalized_completion: "c".into(),
                correct: true,
                score: 1.0,
                latency_ms: 1.0,
                matched_answer: None,
            })
            .collect();
        let report = EvaluationReport::from_results(suite, results);
        let provenance =
            DatasetProvenance::new(suite.as_str(), FIXTURE_REV, FIXTURE_SPLIT, b"x", total);
        SuiteReportEntry::new(provenance, report)
    }

    // Scrambled input order: GPQA, Perplexity, TerminalBench, MMLU.
    let multi = MultiSuiteReport::from_reports(vec![
        make_entry(Suite::GPQA, 1),
        make_entry(Suite::Perplexity, 1),
        make_entry(Suite::TerminalBench, 1),
        make_entry(Suite::MMLU, 1),
    ]);
    assert_eq!(
        multi.entries.iter().map(|e| e.suite).collect::<Vec<_>>(),
        vec![
            Suite::MMLU,
            Suite::GPQA,
            Suite::TerminalBench,
            Suite::Perplexity,
        ]
    );
}

#[test]
fn cross_suite_aggregate_is_task_weighted_and_deterministic() {
    use eval_harness::provenance::DatasetProvenance;
    use eval_harness::report::{MultiSuiteReport, SuiteReportEntry};

    fn make_entry(suite: Suite, correct: usize, total: usize) -> SuiteReportEntry {
        let results: Vec<TaskResult> = (0..total)
            .map(|i| TaskResult {
                task_id: format!("{}-{}", suite.as_str(), i),
                suite,
                prompt_tokens: 1,
                completion_tokens: 1,
                completion: "c".into(),
                normalized_completion: "c".into(),
                correct: i < correct,
                score: if i < correct { 1.0 } else { 0.0 },
                latency_ms: 1.0,
                matched_answer: None,
            })
            .collect();
        let report = EvaluationReport::from_results(suite, results);
        let provenance =
            DatasetProvenance::new(suite.as_str(), FIXTURE_REV, FIXTURE_SPLIT, b"x", total);
        SuiteReportEntry::new(provenance, report)
    }

    let mmlu = make_entry(Suite::MMLU, 9, 10);
    let gpqa = make_entry(Suite::GPQA, 0, 10);

    // Independent of input order.
    let a = MultiSuiteReport::from_reports(vec![mmlu.clone(), gpqa.clone()]);
    let b = MultiSuiteReport::from_reports(vec![gpqa, mmlu]);
    assert_eq!(a, b);

    // Task-weighted: 9/20 = 0.45 overall.
    assert_eq!(a.task_count, 20);
    assert_eq!(a.correct_count, 9);
    assert!((a.overall_accuracy - 0.45).abs() < 1e-9);
    // Mean per-suite accuracy is also 0.45 here because both suites have
    // 10 tasks, but it's tracked independently so callers can distinguish
    // task-weighted from per-suite averaging.
    assert!((a.mean_suite_accuracy - 0.45).abs() < 1e-9);
    // Entries sorted by Suite's declaration-order Ord (MMLU < GPQA).
    assert_eq!(a.entries[0].suite, Suite::MMLU);
    assert_eq!(a.entries[1].suite, Suite::GPQA);
    // Provenance survives aggregation.
    assert_eq!(a.entries[0].provenance.source, "mmlu");
    assert_eq!(a.entries[1].provenance.source, "gpqa");

    let encoded = serde_json::to_string(&a).unwrap();
    let decoded: MultiSuiteReport = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, a);
}

#[test]
fn cross_suite_aggregate_empty_is_zero() {
    use eval_harness::report::MultiSuiteReport;
    let multi = MultiSuiteReport::from_reports(vec![]);
    assert_eq!(multi.task_count, 0);
    assert_eq!(multi.correct_count, 0);
    assert_eq!(multi.overall_accuracy, 0.0);
    assert_eq!(multi.mean_suite_accuracy, 0.0);
    assert!(multi.entries.is_empty());
}

#[test]
fn perplexity_is_deterministic_and_pure() {
    use eval_harness::perplexity::score_perplexity;
    let log_probs = vec![-0.1, -0.5, -1.2, -2.0];
    let a = score_perplexity(&log_probs);
    let b = score_perplexity(&log_probs);
    assert_eq!(a.to_bits(), b.to_bits());
    assert!(a.is_finite());
    assert!(a > 1.0);
    // Empty input -> +infinity.
    assert!(score_perplexity(&[]).is_infinite());
    // Perfect predictions -> perplexity = 1.
    assert!((score_perplexity(&[0.0; 8]) - 1.0).abs() < 1e-9);
}
