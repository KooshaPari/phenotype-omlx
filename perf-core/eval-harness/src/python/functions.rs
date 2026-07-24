//! PyO3 bindings — standalone scoring and convenience functions.

use pyo3::prelude::*;

use super::reports::PyEvaluationReport;
use super::types::{PySuite, PyTaskResult, PyTaskSpec};

/// Normalize an answer for comparison.
#[pyfunction]
pub(super) fn normalize_answer(s: &str) -> String {
    crate::normalize_answer(s)
}

/// Exact-match scoring using normalized comparison.
#[pyfunction]
pub(super) fn score_exact(completion: &str, expected: &str) -> bool {
    crate::score_exact(completion, expected)
}

/// Multiple-choice scoring by letter extraction.
#[pyfunction]
pub(super) fn score_choice(completion: &str, expected: &str, num_choices: usize) -> bool {
    crate::score_choice(completion, expected, num_choices)
}

/// Score a single completion against a task.
#[pyfunction]
pub(super) fn evaluate_task(task: &PyTaskSpec, completion: &str) -> PyResult<PyTaskResult> {
    let rust_task = task.clone().into_rust();
    let result = crate::evaluate(&rust_task, completion)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyTaskResult::from(result))
}

/// Validate an evaluation report: checks internal consistency.
///
/// Returns a list of validation error strings. Empty list = valid.
#[pyfunction]
pub(super) fn validate_report(report: &PyEvaluationReport) -> Vec<String> {
    let mut errors = Vec::new();

    if report.task_count == 0 && report.results.is_empty() {
        return errors;
    }

    if report.task_count != report.results.len() {
        errors.push(format!(
            "task_count ({}) != results.len() ({})",
            report.task_count,
            report.results.len()
        ));
    }

    let actual_correct = report.results.iter().filter(|r| r.correct).count();
    if actual_correct != report.correct_count {
        errors.push(format!(
            "correct_count ({}) != actual correct ({})",
            report.correct_count, actual_correct
        ));
    }

    if report.task_count > 0 {
        let expected_accuracy = actual_correct as f64 / report.task_count as f64;
        if (report.accuracy - expected_accuracy).abs() > 1e-10 {
            errors.push(format!(
                "accuracy ({}) != expected ({})",
                report.accuracy, expected_accuracy
            ));
        }
    } else if report.accuracy != 0.0 {
        errors.push(format!(
            "accuracy ({}) != 0.0 for empty report",
            report.accuracy
        ));
    }

    let actual_mean = if report.task_count == 0 {
        0.0
    } else {
        report.results.iter().map(|r| r.score).sum::<f64>() / report.task_count as f64
    };
    if (report.mean_score - actual_mean).abs() > 1e-10 {
        errors.push(format!(
            "mean_score ({}) != expected ({})",
            report.mean_score, actual_mean
        ));
    }

    for result in &report.results {
        if result.task_id.is_empty() {
            errors.push("empty task_id in results".into());
        }
        if !result.score.is_finite() {
            errors.push(format!("non-finite score for task {}", result.task_id));
        }
        if !result.latency_ms.is_finite() {
            errors.push(format!("non-finite latency_ms for task {}", result.task_id));
        }
    }

    errors
}

/// Deserialize a report from a JSON string and validate it.
///
/// Returns (report, errors) where errors is a list of validation issues.
#[pyfunction]
pub(super) fn ingest_report<'py>(
    py: Python<'py>,
    json: &str,
) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
    let report = PyEvaluationReport::parse_json(json)?;
    let errors = validate_report(&report);
    let report_obj = report.into_pyobject(py)?.into_any();
    let errors_obj = errors.into_pyobject(py)?.into_any();
    pyo3::types::PyTuple::new(py, [report_obj, errors_obj])
}

/// Run a suite summary: accepts pre-computed results and returns aggregated metrics.
///
/// This is a convenience wrapper that builds an EvaluationReport from a list
/// of PyTaskResult objects and validates it.
#[pyfunction]
#[pyo3(signature = (suite, results))]
pub(super) fn run_suite_summary(
    suite: PySuite,
    results: Vec<PyTaskResult>,
) -> PyResult<PyEvaluationReport> {
    let report = PyEvaluationReport::build_from_results(suite, results);
    let errors = validate_report(&report);
    if !errors.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid report: {}",
            errors.join("; ")
        )));
    }
    Ok(report)
}

/// Register all functions on the parent module.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize_answer, m)?)?;
    m.add_function(wrap_pyfunction!(score_exact, m)?)?;
    m.add_function(wrap_pyfunction!(score_choice, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_task, m)?)?;
    m.add_function(wrap_pyfunction!(validate_report, m)?)?;
    m.add_function(wrap_pyfunction!(ingest_report, m)?)?;
    m.add_function(wrap_pyfunction!(run_suite_summary, m)?)?;
    Ok(())
}
