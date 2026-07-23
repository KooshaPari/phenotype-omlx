//! PyO3 bindings for eval-harness supporting Python 3.14 free-threaded environments.

use crate::backend::{Backend, BackendError, Completion, Likelihood};
use crate::dataset::Dataset;
use crate::error::EvalError;
use crate::fixture_backend::OracleBackend;
use crate::provenance::DatasetProvenance;
use crate::report::{MultiSuiteReport, SuiteReportEntry};
use crate::{
    gpqa, mmlu, perplexity, score_choice, score_exact, terminal_bench, EvaluationReport, Suite,
    TaskResult, TaskSpec,
};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

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

fn parse_suite(s: &str) -> PyResult<Suite> {
    match s.trim().to_lowercase().as_str() {
        "mmlu" => Ok(Suite::Mmlu),
        "gpqa" => Ok(Suite::Gpqa),
        "terminal-bench" | "terminal_bench" | "terminalbench" => Ok(Suite::TerminalBench),
        "perplexity" => Ok(Suite::Perplexity),
        _ => Err(PyValueError::new_err(format!(
            "Unknown suite '{s}'. Valid suites: mmlu, gpqa, terminal-bench, perplexity"
        ))),
    }
}

/// Provenance metadata for a loaded evaluation dataset.
#[pyclass(name = "DatasetProvenance", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyDatasetProvenance {
    pub inner: DatasetProvenance,
}

#[pymethods]
impl PyDatasetProvenance {
    #[new]
    #[pyo3(signature = (source, source_revision, split, bytes, task_count))]
    pub fn new(
        source: String,
        source_revision: String,
        split: String,
        bytes: &[u8],
        task_count: usize,
    ) -> Self {
        Self {
            inner: DatasetProvenance::new(source, source_revision, split, bytes, task_count),
        }
    }

    #[getter]
    pub fn source(&self) -> String {
        self.inner.source.clone()
    }

    #[getter]
    pub fn source_revision(&self) -> String {
        self.inner.source_revision.clone()
    }

    #[getter]
    pub fn split(&self) -> String {
        self.inner.split.clone()
    }

    #[getter]
    pub fn sha256(&self) -> String {
        self.inner.sha256.clone()
    }

    #[getter]
    pub fn task_count(&self) -> usize {
        self.inner.task_count
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {e}")))
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("source", &self.inner.source)?;
        dict.set_item("source_revision", &self.inner.source_revision)?;
        dict.set_item("split", &self.inner.split)?;
        dict.set_item("sha256", &self.inner.sha256)?;
        dict.set_item("task_count", self.inner.task_count)?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// A single evaluation task specification.
#[pyclass(name = "TaskSpec", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyTaskSpec {
    pub inner: TaskSpec,
}

#[pymethods]
impl PyTaskSpec {
    #[getter]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    pub fn suite(&self) -> String {
        self.inner.suite.as_str().to_string()
    }

    #[getter]
    pub fn prompt(&self) -> String {
        self.inner.prompt.clone()
    }

    #[getter]
    pub fn expected(&self) -> Option<String> {
        self.inner.expected.clone()
    }

    #[getter]
    pub fn choices(&self) -> Vec<String> {
        self.inner.choices.clone()
    }

    pub fn is_multiple_choice(&self) -> bool {
        self.inner.is_multiple_choice()
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {e}")))
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let json_str = self.to_json()?;
        let json_mod = py.import("json")?;
        let dict = json_mod.call_method1("loads", (json_str,))?;
        dict.cast_into::<PyDict>()
            .map_err(|e| PyValueError::new_err(format!("Dict conversion failed: {e}")))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Evaluation result for a single task.
#[pyclass(name = "TaskResult", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyTaskResult {
    pub inner: TaskResult,
}

#[pymethods]
impl PyTaskResult {
    #[getter]
    pub fn task_id(&self) -> String {
        self.inner.task_id.clone()
    }

    #[getter]
    pub fn suite(&self) -> String {
        self.inner.suite.as_str().to_string()
    }

    #[getter]
    pub fn prompt_tokens(&self) -> usize {
        self.inner.prompt_tokens
    }

    #[getter]
    pub fn completion_tokens(&self) -> usize {
        self.inner.completion_tokens
    }

    #[getter]
    pub fn completion(&self) -> String {
        self.inner.completion.clone()
    }

    #[getter]
    pub fn normalized_completion(&self) -> String {
        self.inner.normalized_completion.clone()
    }

    #[getter]
    pub fn correct(&self) -> bool {
        self.inner.correct
    }

    #[getter]
    pub fn score(&self) -> f64 {
        self.inner.score
    }

    #[getter]
    pub fn latency_ms(&self) -> f64 {
        self.inner.latency_ms
    }

    #[getter]
    pub fn matched_answer(&self) -> Option<String> {
        self.inner.matched_answer.clone()
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {e}")))
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let json_str = self.to_json()?;
        let json_mod = py.import("json")?;
        let dict = json_mod.call_method1("loads", (json_str,))?;
        dict.cast_into::<PyDict>()
            .map_err(|e| PyValueError::new_err(format!("Dict conversion failed: {e}")))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// A loaded dataset with associated provenance and tasks.
#[pyclass(name = "Dataset", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyDataset {
    pub inner: Dataset,
}

#[pymethods]
impl PyDataset {
    #[staticmethod]
    #[pyo3(signature = (suite, path, source_revision=None, split=None))]
    pub fn load(
        suite: &str,
        path: &str,
        source_revision: Option<&str>,
        split: Option<&str>,
    ) -> PyResult<Self> {
        let suite_enum = parse_suite(suite)?;
        let rev = source_revision.unwrap_or("unspecified");
        let sp = split.unwrap_or("unspecified");
        let ds = match suite_enum {
            Suite::Mmlu => mmlu::load_csv_with_provenance(path, rev, sp)?,
            Suite::Gpqa => gpqa::load_jsonl_with_provenance(path, rev, sp)?,
            Suite::TerminalBench => terminal_bench::load_yaml_with_provenance(path, rev, sp)?,
            Suite::Perplexity => {
                return Err(PyValueError::new_err(
                    "Perplexity suite does not use file-based dataset loading",
                ));
            }
        };
        Ok(Self { inner: ds })
    }

    #[staticmethod]
    #[pyo3(signature = (suite, bytes, source="<bytes>", source_revision=None, split=None))]
    pub fn load_bytes(
        suite: &str,
        bytes: &[u8],
        source: &str,
        source_revision: Option<&str>,
        split: Option<&str>,
    ) -> PyResult<Self> {
        let suite_enum = parse_suite(suite)?;
        let rev = source_revision.unwrap_or("unspecified");
        let sp = split.unwrap_or("unspecified");
        let ds = match suite_enum {
            Suite::Mmlu => mmlu::load_csv_bytes(bytes, source, rev, sp)?,
            Suite::Gpqa => gpqa::load_jsonl_bytes(bytes, source, rev, sp)?,
            Suite::TerminalBench => terminal_bench::load_yaml_bytes(bytes, source, rev, sp)?,
            Suite::Perplexity => {
                return Err(PyValueError::new_err(
                    "Perplexity suite does not use file-based dataset loading",
                ));
            }
        };
        Ok(Self { inner: ds })
    }

    #[staticmethod]
    pub fn from_json(json_str: &str) -> PyResult<Self> {
        let ds: Dataset = serde_json::from_str(json_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid dataset JSON: {e}")))?;
        Ok(Self { inner: ds })
    }

    #[getter]
    pub fn suite(&self) -> String {
        self.inner.suite().as_str().to_string()
    }

    #[getter]
    pub fn provenance(&self) -> PyDatasetProvenance {
        PyDatasetProvenance {
            inner: self.inner.provenance().clone(),
        }
    }

    #[getter]
    pub fn tasks(&self) -> Vec<PyTaskSpec> {
        self.inner
            .as_tasks()
            .iter()
            .cloned()
            .map(|t| PyTaskSpec { inner: t })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {e}")))
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let json_str = self.to_json()?;
        let json_mod = py.import("json")?;
        let dict = json_mod.call_method1("loads", (json_str,))?;
        dict.cast_into::<PyDict>()
            .map_err(|e| PyValueError::new_err(format!("Dict conversion failed: {e}")))
    }

    #[pyo3(signature = (backend=None))]
    pub fn run_suite(
        &self,
        py: Python<'_>,
        backend: Option<Py<PyAny>>,
    ) -> PyResult<PyEvaluationReport> {
        py_run_suite(py, self, backend)
    }

    #[pyo3(signature = (backend=None))]
    pub fn run_suite_json(&self, py: Python<'_>, backend: Option<Py<PyAny>>) -> PyResult<String> {
        py_run_suite_json(py, self, backend)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Dataset(suite={}, tasks={})",
            self.inner.suite().as_str(),
            self.inner.len()
        )
    }
}

/// Evaluation report containing aggregated per-suite task results.
#[pyclass(name = "EvaluationReport", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyEvaluationReport {
    pub inner: EvaluationReport,
}

#[pymethods]
impl PyEvaluationReport {
    #[staticmethod]
    pub fn from_json(json_str: &str) -> PyResult<Self> {
        let report: EvaluationReport = serde_json::from_str(json_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid EvaluationReport JSON: {e}")))?;
        Ok(Self { inner: report })
    }

    #[getter]
    pub fn suite(&self) -> String {
        self.inner.suite.as_str().to_string()
    }

    #[getter]
    pub fn task_count(&self) -> usize {
        self.inner.task_count
    }

    #[getter]
    pub fn correct_count(&self) -> usize {
        self.inner.correct_count
    }

    #[getter]
    pub fn accuracy(&self) -> f64 {
        self.inner.accuracy
    }

    #[getter]
    pub fn mean_score(&self) -> f64 {
        self.inner.mean_score
    }

    #[getter]
    pub fn results(&self) -> Vec<PyTaskResult> {
        self.inner
            .results
            .iter()
            .cloned()
            .map(|r| PyTaskResult { inner: r })
            .collect()
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {e}")))
    }

    pub fn to_json_pretty(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {e}")))
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let json_str = self.to_json()?;
        let json_mod = py.import("json")?;
        let dict = json_mod.call_method1("loads", (json_str,))?;
        dict.cast_into::<PyDict>()
            .map_err(|e| PyValueError::new_err(format!("Dict conversion failed: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "EvaluationReport(suite={}, tasks={}, accuracy={:.4})",
            self.inner.suite.as_str(),
            self.inner.task_count,
            self.inner.accuracy
        )
    }

    fn __str__(&self) -> PyResult<String> {
        self.to_json()
    }
}

/// Cross-suite aggregate evaluation report.
#[pyclass(name = "MultiSuiteReport", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMultiSuiteReport {
    pub inner: MultiSuiteReport,
}

#[pymethods]
impl PyMultiSuiteReport {
    #[staticmethod]
    pub fn from_reports(
        reports: Vec<PyEvaluationReport>,
        provenances: Vec<PyDatasetProvenance>,
    ) -> PyResult<Self> {
        if reports.len() != provenances.len() {
            return Err(PyValueError::new_err(
                "reports and provenances must have matching lengths",
            ));
        }
        let entries: Vec<SuiteReportEntry> = reports
            .into_iter()
            .zip(provenances)
            .map(|(r, p)| SuiteReportEntry::new(p.inner, r.inner))
            .collect();
        Ok(Self {
            inner: MultiSuiteReport::from_reports(entries),
        })
    }

    #[staticmethod]
    pub fn from_json(json_str: &str) -> PyResult<Self> {
        let multi: MultiSuiteReport = serde_json::from_str(json_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid MultiSuiteReport JSON: {e}")))?;
        Ok(Self { inner: multi })
    }

    #[getter]
    pub fn task_count(&self) -> usize {
        self.inner.task_count
    }

    #[getter]
    pub fn correct_count(&self) -> usize {
        self.inner.correct_count
    }

    #[getter]
    pub fn overall_accuracy(&self) -> f64 {
        self.inner.overall_accuracy
    }

    #[getter]
    pub fn mean_suite_accuracy(&self) -> f64 {
        self.inner.mean_suite_accuracy
    }

    #[getter]
    pub fn mean_suite_score(&self) -> f64 {
        self.inner.mean_suite_score
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {e}")))
    }

    pub fn to_json_pretty(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {e}")))
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let json_str = self.to_json()?;
        let json_mod = py.import("json")?;
        let dict = json_mod.call_method1("loads", (json_str,))?;
        dict.cast_into::<PyDict>()
            .map_err(|e| PyValueError::new_err(format!("Dict conversion failed: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiSuiteReport(tasks={}, overall_accuracy={:.4})",
            self.inner.task_count, self.inner.overall_accuracy
        )
    }

    fn __str__(&self) -> PyResult<String> {
        self.to_json()
    }
}

/// Fixture oracle backend for deterministic testing.
#[pyclass(name = "OracleBackend", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyOracleBackend {
    pub tasks: Arc<Vec<TaskSpec>>,
}

#[pymethods]
impl PyOracleBackend {
    #[new]
    pub fn new(dataset: &PyDataset) -> Self {
        Self {
            tasks: Arc::new(dataset.inner.as_tasks().to_vec()),
        }
    }
}

struct PythonBackendWrapper {
    obj: Py<PyAny>,
}

impl Backend for PythonBackendWrapper {
    fn complete(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> std::result::Result<Completion, BackendError> {
        Python::attach(|py| {
            let py_obj = self.obj.bind(py);
            let res = if py_obj.hasattr("complete").unwrap_or(false) {
                py_obj.call_method1("complete", (prompt, max_tokens))
            } else if py_obj.is_callable() {
                py_obj.call1((prompt, max_tokens))
            } else {
                return Err(BackendError::InvalidResponse {
                    message: "Backend object has no 'complete' method and is not callable".into(),
                });
            };

            let res = res.map_err(|e| BackendError::Unavailable {
                message: e.to_string(),
            })?;

            if let Ok(text) = res.extract::<String>() {
                let prompt_tokens = prompt.split_whitespace().count();
                let completion_tokens = text.split_whitespace().count();
                Ok(Completion {
                    text,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms: 0.0,
                })
            } else if let Ok(dict) = res.cast::<PyDict>() {
                let text: String = match dict.get_item("text") {
                    Ok(Some(val)) => {
                        val.extract()
                            .map_err(|e: pyo3::PyErr| BackendError::InvalidResponse {
                                message: e.to_string(),
                            })?
                    }
                    _ => {
                        return Err(BackendError::InvalidResponse {
                            message: "missing 'text'".into(),
                        })
                    }
                };

                let prompt_tokens: usize = dict
                    .get_item("prompt_tokens")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_else(|| prompt.split_whitespace().count());

                let completion_tokens: usize = dict
                    .get_item("completion_tokens")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_else(|| text.split_whitespace().count());

                let latency_ms: f64 = dict
                    .get_item("latency_ms")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0.0);

                Ok(Completion {
                    text,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                })
            } else if let Ok((text, prompt_tokens, completion_tokens, latency_ms)) =
                res.extract::<(String, usize, usize, f64)>()
            {
                Ok(Completion {
                    text,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                })
            } else {
                Err(BackendError::InvalidResponse {
                    message: format!("unexpected return type from backend complete: {res}"),
                })
            }
        })
    }

    fn log_likelihood(
        &self,
        prompt: &str,
        continuation: &str,
    ) -> std::result::Result<Likelihood, BackendError> {
        Python::attach(|py| {
            let py_obj = self.obj.bind(py);
            let res = if py_obj.hasattr("log_likelihood").unwrap_or(false) {
                py_obj.call_method1("log_likelihood", (prompt, continuation))
            } else if py_obj.hasattr("log_prob").unwrap_or(false) {
                py_obj.call_method1("log_prob", (prompt, continuation))
            } else {
                let comp = self.complete(prompt, continuation.split_whitespace().count().max(1))?;
                let log_prob = if comp.text.trim() == continuation.trim() {
                    0.0
                } else {
                    -10.0
                };
                return Ok(Likelihood {
                    log_probability: log_prob,
                    token_count: comp.completion_tokens,
                    latency_ms: comp.latency_ms,
                });
            };

            let res = res.map_err(|e| BackendError::Unavailable {
                message: e.to_string(),
            })?;

            if let Ok(log_probability) = res.extract::<f64>() {
                let token_count = continuation.split_whitespace().count();
                Ok(Likelihood {
                    log_probability,
                    token_count,
                    latency_ms: 0.0,
                })
            } else if let Ok(dict) = res.cast::<PyDict>() {
                let prob_val = match dict.get_item("log_probability") {
                    Ok(Some(val)) => Some(val),
                    _ => match dict.get_item("log_prob") {
                        Ok(Some(val)) => Some(val),
                        _ => None,
                    },
                };
                let log_probability: f64 = match prob_val {
                    Some(val) => {
                        val.extract()
                            .map_err(|e: pyo3::PyErr| BackendError::InvalidResponse {
                                message: e.to_string(),
                            })?
                    }
                    None => {
                        return Err(BackendError::InvalidResponse {
                            message: "missing 'log_probability'".into(),
                        })
                    }
                };

                let token_count: usize = dict
                    .get_item("token_count")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_else(|| continuation.split_whitespace().count());

                let latency_ms: f64 = dict
                    .get_item("latency_ms")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0.0);

                Ok(Likelihood {
                    log_probability,
                    token_count,
                    latency_ms,
                })
            } else if let Ok((log_probability, token_count, latency_ms)) =
                res.extract::<(f64, usize, f64)>()
            {
                Ok(Likelihood {
                    log_probability,
                    token_count,
                    latency_ms,
                })
            } else {
                Err(BackendError::InvalidResponse {
                    message: format!("unexpected return type from backend log_likelihood: {res}"),
                })
            }
        })
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

    let results = py.detach(|| -> Result<Vec<TaskResult>, EvalError> {
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
