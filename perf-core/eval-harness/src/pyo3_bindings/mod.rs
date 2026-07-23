//! PyO3 bindings for eval-harness supporting Python 3.14 free-threaded environments.

use crate::error::EvalError;
use crate::fixture_backend::OracleBackend;
use crate::{perplexity, score_choice, score_exact, EvaluationReport, TaskResult};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

mod evaluation;
mod scoring;
mod task;

pub use evaluation::*;
pub(crate) use scoring::PythonBackendWrapper;
pub use task::*;

impl From<EvalError> for PyErr {
    fn from(err: EvalError) -> Self {
        match err {
            EvalError::Io { path, source } => {
                PyIOError::new_err(format!("IO error for {path}: {source}"))
            }
            EvalError::Backend { task_id, source } => {
                PyRuntimeError::new_err(format!("Backend error for {task_id}: {source}"))
            }
            other => PyValueError::new_err(other.to_string()),
        }
    }
}

pub(crate) fn parse_suite(s: &str) -> PyResult<crate::Suite> {
    match s.trim().to_lowercase().as_str() {
        "mmlu" => Ok(crate::Suite::Mmlu),
        "gpqa" => Ok(crate::Suite::Gpqa),
        "terminal-bench" | "terminal_bench" | "terminalbench" => Ok(crate::Suite::TerminalBench),
        "perplexity" => Ok(crate::Suite::Perplexity),
        _ => Err(PyValueError::new_err(format!(
            "Unknown suite '{s}'. Valid suites: mmlu, gpqa, terminal-bench, perplexity"
        ))),
    }
}

/// Load a dataset from a file path.
#[pyfunction]
#[pyo3(signature = (suite, path, source_revision=None, split=None))]
pub fn py_load_dataset(
    suite: &str,
    path: &str,
    source_revision: Option<&str>,
    split: Option<&str>,
) -> PyResult<PyDataset> {
    PyDataset::load(suite, path, source_revision, split)
}

/// Load a dataset from raw bytes.
#[pyfunction]
#[pyo3(signature = (suite, bytes, source="<bytes>", source_revision=None, split=None))]
pub fn py_load_dataset_bytes(
    suite: &str,
    bytes: &[u8],
    source: &str,
    source_revision: Option<&str>,
    split: Option<&str>,
) -> PyResult<PyDataset> {
    PyDataset::load_bytes(suite, bytes, source, source_revision, split)
}

/// Load MMLU dataset from CSV file path.
#[pyfunction]
#[pyo3(signature = (path, source_revision=None, split=None))]
pub fn py_load_mmlu_csv(
    path: &str,
    source_revision: Option<&str>,
    split: Option<&str>,
) -> PyResult<PyDataset> {
    PyDataset::load("mmlu", path, source_revision, split)
}

/// Load MMLU dataset from CSV bytes.
#[pyfunction]
#[pyo3(signature = (bytes, source="<bytes>", source_revision=None, split=None))]
pub fn py_load_mmlu_csv_bytes(
    bytes: &[u8],
    source: &str,
    source_revision: Option<&str>,
    split: Option<&str>,
) -> PyResult<PyDataset> {
    PyDataset::load_bytes("mmlu", bytes, source, source_revision, split)
}

/// Load GPQA dataset from JSONL file path.
#[pyfunction]
#[pyo3(signature = (path, source_revision=None, split=None))]
pub fn py_load_gpqa_jsonl(
    path: &str,
    source_revision: Option<&str>,
    split: Option<&str>,
) -> PyResult<PyDataset> {
    PyDataset::load("gpqa", path, source_revision, split)
}

/// Load GPQA dataset from JSONL bytes.
#[pyfunction]
#[pyo3(signature = (bytes, source="<bytes>", source_revision=None, split=None))]
pub fn py_load_gpqa_jsonl_bytes(
    bytes: &[u8],
    source: &str,
    source_revision: Option<&str>,
    split: Option<&str>,
) -> PyResult<PyDataset> {
    PyDataset::load_bytes("gpqa", bytes, source, source_revision, split)
}

/// Load TerminalBench dataset from YAML file path.
#[pyfunction]
#[pyo3(signature = (path, source_revision=None, split=None))]
pub fn py_load_terminal_bench_yaml(
    path: &str,
    source_revision: Option<&str>,
    split: Option<&str>,
) -> PyResult<PyDataset> {
    PyDataset::load("terminal-bench", path, source_revision, split)
}

/// Load TerminalBench dataset from YAML bytes.
#[pyfunction]
#[pyo3(signature = (bytes, source="<bytes>", source_revision=None, split=None))]
pub fn py_load_terminal_bench_yaml_bytes(
    bytes: &[u8],
    source: &str,
    source_revision: Option<&str>,
    split: Option<&str>,
) -> PyResult<PyDataset> {
    PyDataset::load_bytes("terminal-bench", bytes, source, source_revision, split)
}

/// Run an evaluation suite on a dataset and return an EvaluationReport object.
#[pyfunction]
#[pyo3(signature = (dataset, backend=None))]
pub fn py_run_suite(
    py: Python<'_>,
    dataset: &PyDataset,
    backend: Option<Py<PyAny>>,
) -> PyResult<PyEvaluationReport> {
    let suite = dataset.inner.suite();
    let tasks = dataset.inner.as_tasks();

    let results = py.detach(|| -> std::result::Result<Vec<TaskResult>, EvalError> {
        if let Some(be) = backend {
            Python::attach(|py_inner| {
                if be.bind(py_inner).is_instance_of::<PyOracleBackend>() {
                    let oracle = OracleBackend::new(tasks);
                    crate::runner::run_suite(suite, &oracle, tasks)
                } else {
                    let py_be = PythonBackendWrapper { obj: be };
                    crate::runner::run_suite(suite, &py_be, tasks)
                }
            })
        } else {
            let oracle = OracleBackend::new(tasks);
            crate::runner::run_suite(suite, &oracle, tasks)
        }
    })?;

    let report = EvaluationReport::from_results(suite, results);
    Ok(PyEvaluationReport { inner: report })
}

/// Run an evaluation suite on a dataset and return EvaluationReport JSON string.
#[pyfunction]
#[pyo3(signature = (dataset, backend=None))]
pub fn py_run_suite_json(
    py: Python<'_>,
    dataset: &PyDataset,
    backend: Option<Py<PyAny>>,
) -> PyResult<String> {
    let report = py_run_suite(py, dataset, backend)?;
    report.to_json()
}

/// Compute perplexity from token log-probabilities.
#[pyfunction]
pub fn py_score_perplexity(log_probs: Vec<f64>) -> f64 {
    perplexity::score_perplexity(&log_probs)
}

/// Normalize an answer string for exact match comparison.
#[pyfunction]
pub fn py_normalize_answer(s: &str) -> String {
    crate::normalize_answer(s)
}

/// Check exact match between completion and expected answer.
#[pyfunction]
pub fn py_score_exact(completion: &str, expected: &str) -> bool {
    score_exact(completion, expected)
}

/// Check multiple-choice option selection.
#[pyfunction]
pub fn py_score_choice(completion: &str, expected: &str, num_choices: usize) -> bool {
    score_choice(completion, expected, num_choices)
}

/// Define the Python module `eval_harness`.
#[pymodule]
pub fn eval_harness(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatasetProvenance>()?;
    m.add_class::<PyTaskSpec>()?;
    m.add_class::<PyTaskResult>()?;
    m.add_class::<PyDataset>()?;
    m.add_class::<PyEvaluationReport>()?;
    m.add_class::<PyMultiSuiteReport>()?;
    m.add_class::<PyOracleBackend>()?;

    m.add_function(wrap_pyfunction!(py_load_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_dataset_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_mmlu_csv, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_mmlu_csv_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_gpqa_jsonl, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_gpqa_jsonl_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_terminal_bench_yaml, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_terminal_bench_yaml_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_suite, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_suite_json, m)?)?;
    m.add_function(wrap_pyfunction!(py_score_perplexity, m)?)?;
    m.add_function(wrap_pyfunction!(py_normalize_answer, m)?)?;
    m.add_function(wrap_pyfunction!(py_score_exact, m)?)?;
    m.add_function(wrap_pyfunction!(py_score_choice, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CSV: &str = "subject,question,A,B,C,D,answer\n\
        anatomy,The heart is located in which cavity?,Cranial,Thoracic,Abdominal,Pelvic,B\n";

    const SAMPLE_GPQA: &str = r#"{"id":"chemistry-1","question":"What is H2O?","choices":["Water","Gold","Air","Fire"],"answer":"A"}"#;
    const SAMPLE_TB: &str =
        "id: test-task\nprompt: echo hello\ncriteria:\n  expected_commands: [\"echo\"]\n";

    #[test]
    fn test_py_dataset_loading_and_suite_run() {
        Python::initialize();
        Python::attach(|py| {
            let dataset = PyDataset::load_bytes(
                "mmlu",
                SAMPLE_CSV.as_bytes(),
                "test.csv",
                Some("v1"),
                Some("test"),
            )
            .expect("Dataset loading failed");

            assert_eq!(dataset.suite(), "mmlu");
            assert_eq!(dataset.len(), 1);

            let report = py_run_suite(py, &dataset, None).expect("Run suite failed");
            assert_eq!(report.suite(), "mmlu");
            assert_eq!(report.task_count(), 1);
            assert_eq!(report.correct_count(), 1);
            assert_eq!(report.accuracy(), 1.0);

            let json_str = py_run_suite_json(py, &dataset, None).expect("Run suite json failed");
            assert!(json_str.contains("\"accuracy\":1.0"));
            assert!(json_str.contains("\"suite\":\"mmlu\""));

            let py_report =
                PyEvaluationReport::from_json(&json_str).expect("Report deserialization failed");
            assert_eq!(py_report.suite(), "mmlu");
        });
    }

    #[test]
    fn test_py_gpqa_and_terminal_bench() {
        Python::initialize();
        Python::attach(|py| {
            let gpqa_ds = py_load_gpqa_jsonl_bytes(
                SAMPLE_GPQA.as_bytes(),
                "gpqa.jsonl",
                Some("v1"),
                Some("main"),
            )
            .expect("GPQA load failed");
            assert_eq!(gpqa_ds.suite(), "gpqa");
            assert_eq!(gpqa_ds.len(), 1);

            let tb_ds = py_load_terminal_bench_yaml_bytes(
                SAMPLE_TB.as_bytes(),
                "tb.yaml",
                Some("v1"),
                Some("test"),
            )
            .expect("TB load failed");
            assert_eq!(tb_ds.suite(), "terminal-bench");
            assert_eq!(tb_ds.len(), 1);

            let report = tb_ds.run_suite(py, None).expect("TB run suite failed");
            assert_eq!(report.suite(), "terminal-bench");
        });
    }
}
